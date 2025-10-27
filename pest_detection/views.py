import io
import os
import time
import uuid
import logging
from typing import Dict, Any, List
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.contrib.auth.models import User

from rest_framework import status, viewsets
from rest_framework.decorators import api_view, action, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

from PIL import Image
from ultralytics import YOLO

from .models import (
    PestType, DetectionModel, DetectionSession, PestDetection,
    PestCount, DetectionFeedback, DetectionAnalytics
)
from .serializers import (
    PestTypeSerializer, DetectionModelSerializer, DetectionSessionSerializer,
    PestDetectionSerializer, PestCountSerializer, DetectionFeedbackSerializer,
    PestDetectionCreateSerializer, PestDetectionResultSerializer,
    HealthCheckSerializer, DetectionStatsSerializer, BulkDetectionSerializer
)

logger = logging.getLogger('pest_detection')

# Global model cache
_model_cache = None


def load_yolo_model():
    """Load YOLO model for pest detection"""
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    model_path = settings.ML_SETTINGS['YOLO_MODEL_PATH']
    
    # Try to find model file in various locations
    possible_paths = [
        model_path,
        Path(settings.BASE_DIR) / 'pest_detection_ml' / 'models' / 'best.pt',
        Path(settings.BASE_DIR) / 'ml_models' / 'pest_detection' / 'best.pt',
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            logger.info(f"Loading YOLO model from: {path}")
            _model_cache = YOLO(str(path))
            return _model_cache
    
    # If no model found, try to get the active model from database
    try:
        active_model = DetectionModel.objects.filter(is_active=True).first()
        if active_model and active_model.model_path and Path(active_model.model_path).exists():
            logger.info(f"Loading active model from database: {active_model.model_path}")
            _model_cache = YOLO(active_model.model_path)
            return _model_cache
    except Exception as e:
        logger.error(f"Failed to load model from database: {e}")
    
    raise FileNotFoundError(f"No YOLO model found. Checked paths: {possible_paths}")


def aggregate_pest_counts(results, confidence_thresholds: Dict[str, float] = None) -> Dict[str, int]:
    """Aggregate pest counts from YOLO detection results"""
    # Get class names from database instead of hardcoded settings
    class_names = [pest.name for pest in PestType.objects.all().order_by('name')]
    counts = {name: 0 for name in class_names}
    
    if not results or not results[0] or not hasattr(results[0], 'boxes'):
        return counts
    
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return counts
    
    # Get confidence thresholds from database instead of hardcoded values
    default_thresholds = {pest.name: pest.confidence_threshold for pest in PestType.objects.all()}
    
    thresholds = confidence_thresholds or default_thresholds
    
    try:
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item() if hasattr(boxes.cls, 'item') else boxes.cls.tolist()[i])
            conf = float(boxes.conf[i].item() if hasattr(boxes.conf, 'item') else boxes.conf.tolist()[i])
            
            if 0 <= cls_idx < len(class_names):
                pest_name = class_names[cls_idx]
                threshold = thresholds.get(pest_name, 0.25)
                
                if conf >= threshold:
                    counts[pest_name] += 1
    except Exception as e:
        logger.error(f"Error processing detection results: {e}")
    
    return counts


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint - equivalent to Flask /health"""
    try:
        # Check database connection
        db_status = "ok"
        try:
            User.objects.first()
        except Exception:
            db_status = "error"
        
        # Check if model can be loaded
        model_loaded = False
        try:
            model = load_yolo_model()
            model_loaded = True
            model_name = "YOLO Model Loaded"
        except Exception as e:
            model_name = f"Model Error: {str(e)}"
        
        # Get detection statistics
        total_detections = PestDetection.objects.count()
        active_sessions = DetectionSession.objects.filter(
            created_at__gte=timezone.now() - timezone.timedelta(hours=24)
        ).count()
        
        data = {
            'status': 'ok' if db_status == 'ok' and model_loaded else 'warning',
            'model': model_name,
            'classes': [pest.name for pest in PestType.objects.all().order_by('name')],
            'database_status': db_status,
            'model_loaded': model_loaded,
            'total_detections': total_detections,
            'active_sessions': active_sessions
        }
        
        serializer = HealthCheckSerializer(data)
        return Response(serializer.data)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response({
            'status': 'error',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PestDetectionAPIView(APIView):
    """Main pest detection API - equivalent to Flask /detect"""
    
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [AllowAny]  # Allow unauthenticated access for mobile apps
    
    def post(self, request):
        serializer = PestDetectionCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        try:
            # Load model
            model = load_yolo_model()
            
            # Process image
            image_file = data['image']
            image_bytes = image_file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Get or create detection session
            session = self.get_or_create_session(request, data)
            
            # Run inference
            start_time = time.time()
            results = model.predict(
                img,
                imgsz=data.get('image_size', 512),
                conf=data.get('confidence_threshold', 0.25),
                iou=data.get('iou_threshold', 0.50),
                device="cpu"
            )
            inference_time = (time.time() - start_time) * 1000
            
            # Process results
            confidence_thresholds = self.get_confidence_thresholds(data)
            pest_counts = aggregate_pest_counts(results, confidence_thresholds)
            
            # Save detection to database
            detection = self.save_detection(
                session, image_file, img, pest_counts, 
                results, inference_time, data
            )
            
            # Generate response
            response_data = self.generate_response(
                detection, pest_counts, inference_time, data.get('debug', False), results
            )
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def get_or_create_session(self, request, data):
        """Get or create a detection session"""
        # Try to get session from request headers or create new one
        session_id = request.headers.get('X-Session-ID')
        
        if session_id:
            try:
                session = DetectionSession.objects.get(session_id=session_id)
                return session
            except DetectionSession.DoesNotExist:
                pass
        
        # Create new session
        active_model = DetectionModel.objects.filter(is_active=True).first()
        session = DetectionSession.objects.create(
            user=request.user if request.user.is_authenticated else None,
            model_used=active_model,
            confidence_threshold=data.get('confidence_threshold', 0.25),
            iou_threshold=data.get('iou_threshold', 0.50),
            image_size=data.get('image_size', 512)
        )
        return session
    
    def get_confidence_thresholds(self, data):
        """Get confidence thresholds for different pest types"""
        # Get thresholds from database instead of hardcoded values
        thresholds = {pest.name: pest.confidence_threshold for pest in PestType.objects.all()}
        
        # Handle special case for black-bug disable flag (if needed)
        if data.get('disable_black_bug', False) and 'black-bug' in thresholds:
            thresholds['black-bug'] = 1.0
        
        return thresholds
    
    @transaction.atomic
    def save_detection(self, session, image_file, img, pest_counts, results, inference_time, data):
        """Save detection results to database"""
        # Create PestDetection record
        detection = PestDetection.objects.create(
            session=session,
            image=image_file,
            image_name=image_file.name,
            image_size_width=img.width,
            image_size_height=img.height,
            total_pests_detected=sum(pest_counts.values()),
            detection_results=pest_counts,
            raw_detections=self.extract_raw_detections(results),
            inference_time_ms=inference_time,
            latitude=data.get('latitude'),
            longitude=data.get('longitude'),
            farm_location=data.get('farm_location', '')
        )
        
        # Create PestCount records
        for pest_name, count in pest_counts.items():
            if count > 0:
                pest_type = PestType.objects.filter(name=pest_name).first()
                if pest_type:
                    PestCount.objects.create(
                        detection=detection,
                        pest_type=pest_type,
                        count=count,
                        confidence_avg=self.calculate_avg_confidence(results, pest_name)
                    )
        
        return detection
    
    def extract_raw_detections(self, results):
        """Extract raw detection data from YOLO results"""
        if not results or not results[0] or not hasattr(results[0], 'boxes'):
            return []
        
        raw_detections = []
        boxes = results[0].boxes
        class_names = [pest.name for pest in PestType.objects.all().order_by('name')]
        
        if boxes is not None:
            try:
                for i in range(len(boxes)):
                    cls_idx = int(boxes.cls[i].item() if hasattr(boxes.cls, 'item') else boxes.cls.tolist()[i])
                    conf = float(boxes.conf[i].item() if hasattr(boxes.conf, 'item') else boxes.conf.tolist()[i])
                    
                    raw_detections.append({
                        'class_id': cls_idx,
                        'class_name': class_names[cls_idx] if 0 <= cls_idx < len(class_names) else 'unknown',
                        'confidence': round(conf, 3),
                        'bbox': boxes.xyxy[i].tolist() if hasattr(boxes, 'xyxy') else None
                    })
            except Exception as e:
                logger.error(f"Error extracting raw detections: {e}")
        
        return raw_detections
    
    def calculate_avg_confidence(self, results, pest_name):
        """Calculate average confidence for a specific pest type"""
        if not results or not results[0] or not hasattr(results[0], 'boxes'):
            return None
        
        boxes = results[0].boxes
        class_names = [pest.name for pest in PestType.objects.all().order_by('name')]
        
        if pest_name not in class_names or boxes is None:
            return None
        
        target_class_id = class_names.index(pest_name)
        confidences = []
        
        try:
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i].item() if hasattr(boxes.cls, 'item') else boxes.cls.tolist()[i])
                if cls_idx == target_class_id:
                    conf = float(boxes.conf[i].item() if hasattr(boxes.conf, 'item') else boxes.conf.tolist()[i])
                    confidences.append(conf)
        except Exception as e:
            logger.error(f"Error calculating average confidence: {e}")
        
        return sum(confidences) / len(confidences) if confidences else None
    
    def generate_response(self, detection, pest_counts, inference_time, debug_mode, results):
        """Generate API response"""
        # Get pest information
        pest_diseases = {}
        pest_recommendations = {}
        
        for pest_name, count in pest_counts.items():
            if count > 0:
                try:
                    pest_type = PestType.objects.get(name=pest_name)
                    pest_diseases[pest_name] = pest_type.diseases_caused or []
                    pest_recommendations[pest_name] = pest_type.pesticide_recommendations or ""
                except PestType.DoesNotExist:
                    continue
        
        response_data = {
            'status': 'success',
            'session_id': detection.session.session_id,
            'detection_id': detection.id,
            'pest_counts': pest_counts,
            'diseases': pest_diseases,
            'recommendations': pest_recommendations,
            'inference_time_ms': round(inference_time, 1),
            'model_used': detection.session.model_used.name if detection.session.model_used else 'YOLO'
        }
        
        # Add debug information if requested
        if debug_mode:
            response_data['detections'] = detection.raw_detections
            response_data['raw_results'] = {
                'total_detections': len(detection.raw_detections),
                'image_size': f"{detection.image_size_width}x{detection.image_size_height}",
                'session_config': {
                    'confidence_threshold': detection.session.confidence_threshold,
                    'iou_threshold': detection.session.iou_threshold,
                    'image_size': detection.session.image_size
                }
            }
        
        return response_data


@api_view(['GET'])
@permission_classes([AllowAny])
def detection_stats(request):
    """Get detection statistics"""
    try:
        # Basic stats
        total_detections = PestDetection.objects.count()
        total_sessions = DetectionSession.objects.count()
        total_pests = PestDetection.objects.aggregate(
            total=models.Sum('total_pests_detected')
        )['total'] or 0
        
        # Recent activity
        now = timezone.now()
        detections_24h = PestDetection.objects.filter(
            processed_at__gte=now - timezone.timedelta(hours=24)
        ).count()
        detections_7d = PestDetection.objects.filter(
            processed_at__gte=now - timezone.timedelta(days=7)
        ).count()
        detections_30d = PestDetection.objects.filter(
            processed_at__gte=now - timezone.timedelta(days=30)
        ).count()
        
        # Pest type statistics
        pest_type_stats = {}
        for pest_type in PestType.objects.all():
            count = PestCount.objects.filter(pest_type=pest_type).aggregate(
                total=models.Sum('count')
            )['total'] or 0
            pest_type_stats[pest_type.name] = count
        
        # Performance stats
        avg_inference_time = PestDetection.objects.aggregate(
            avg=models.Avg('inference_time_ms')
        )['avg'] or 0
        
        # Location stats
        unique_locations = PestDetection.objects.exclude(
            farm_location=''
        ).values('farm_location').distinct().count()
        
        most_active_location = PestDetection.objects.exclude(
            farm_location=''
        ).values('farm_location').annotate(
            count=models.Count('id')
        ).order_by('-count').first()
        
        data = {
            'total_detections': total_detections,
            'total_sessions': total_sessions,
            'total_pests_detected': total_pests,
            'pest_type_stats': pest_type_stats,
            'detections_last_24h': detections_24h,
            'detections_last_7d': detections_7d,
            'detections_last_30d': detections_30d,
            'avg_inference_time_ms': round(avg_inference_time, 2),
            'avg_confidence': 0.0,  # TODO: Calculate from raw detections
            'unique_locations': unique_locations,
            'most_active_location': most_active_location['farm_location'] if most_active_location else ''
        }
        
        serializer = DetectionStatsSerializer(data)
        return Response(serializer.data)
        
    except Exception as e:
        logger.error(f"Stats calculation failed: {e}")
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ViewSets for CRUD operations
class PestTypeViewSet(viewsets.ModelViewSet):
    """ViewSet for managing pest types"""
    
    queryset = PestType.objects.all()
    serializer_class = PestTypeSerializer
    permission_classes = [IsAuthenticated]


class DetectionModelViewSet(viewsets.ModelViewSet):
    """ViewSet for managing detection models"""
    
    queryset = DetectionModel.objects.all()
    serializer_class = DetectionModelSerializer
    permission_classes = [IsAuthenticated]
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Activate a specific model"""
        model = self.get_object()
        
        # Deactivate all other models
        DetectionModel.objects.update(is_active=False)
        
        # Activate this model
        model.is_active = True
        model.save()
        
        # Clear model cache to force reload
        global _model_cache
        _model_cache = None
        
        return Response({'status': 'Model activated successfully'})


class DetectionSessionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing detection sessions"""
    
    queryset = DetectionSession.objects.all()
    serializer_class = DetectionSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        if not self.request.user.is_staff:
            queryset = queryset.filter(user=self.request.user)
        return queryset


class PestDetectionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing pest detections"""
    
    queryset = PestDetection.objects.all()
    serializer_class = PestDetectionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        if not self.request.user.is_staff:
            queryset = queryset.filter(session__user=self.request.user)
        return queryset
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get recent detections"""
        days = int(request.query_params.get('days', 7))
        since = timezone.now() - timezone.timedelta(days=days)
        
        detections = self.get_queryset().filter(
            processed_at__gte=since
        ).order_by('-processed_at')[:50]
        
        serializer = self.get_serializer(detections, many=True)
        return Response(serializer.data)


class DetectionFeedbackViewSet(viewsets.ModelViewSet):
    """ViewSet for managing detection feedback"""
    
    queryset = DetectionFeedback.objects.all()
    serializer_class = DetectionFeedbackSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
