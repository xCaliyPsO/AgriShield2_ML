import os
import logging
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.db.models import Q, Count, Avg, Sum
from django.shortcuts import get_object_or_404
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from rest_framework import status, viewsets
from rest_framework.decorators import api_view, action, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

from .models import (
    Dataset, TrainingJob, TrainingMetric, TrainingLog, ModelCheckpoint,
    TrainingImage, TrainingReport
)
from .serializers import (
    DatasetSerializer, TrainingJobSerializer, TrainingMetricSerializer,
    TrainingLogSerializer, ModelCheckpointSerializer, TrainingImageSerializer,
    TrainingReportSerializer, TrainingJobCreateSerializer, TrainingJobStatusSerializer,
    TrainingStatsSerializer, BulkImageUploadSerializer
)

logger = logging.getLogger('ml_training')


class DatasetViewSet(viewsets.ModelViewSet):
    """ViewSet for managing training datasets"""
    
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @action(detail=True, methods=['post'])
    def prepare(self, request, pk=None):
        """Prepare dataset by organizing images and creating splits"""
        dataset = self.get_object()
        
        try:
            # Update dataset status
            dataset.status = 'processing'
            dataset.save()
            
            # Count images by type
            train_count = TrainingImage.objects.filter(dataset=dataset, image_type='train').count()
            val_count = TrainingImage.objects.filter(dataset=dataset, image_type='val').count()
            test_count = TrainingImage.objects.filter(dataset=dataset, image_type='test').count()
            total_count = train_count + val_count + test_count
            
            # Update dataset statistics
            dataset.train_images = train_count
            dataset.val_images = val_count
            dataset.test_images = test_count
            dataset.total_images = total_count
            
            # Calculate class distribution
            class_distribution = {}
            for image in TrainingImage.objects.filter(dataset=dataset):
                if image.annotations_json:
                    # Count objects per class in annotations
                    annotations = image.annotations_json
                    if 'objects' in annotations:
                        for obj in annotations['objects']:
                            class_name = obj.get('class', 'unknown')
                            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
            
            dataset.class_distribution = class_distribution
            dataset.class_names = list(class_distribution.keys())
            dataset.status = 'ready'
            dataset.save()
            
            return Response({
                'status': 'success',
                'message': 'Dataset prepared successfully',
                'statistics': {
                    'total_images': total_count,
                    'train_images': train_count,
                    'val_images': val_count,
                    'test_images': test_count,
                    'class_distribution': class_distribution
                }
            })
            
        except Exception as e:
            dataset.status = 'error'
            dataset.save()
            logger.error(f"Dataset preparation failed: {e}")
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload_images(self, request, pk=None):
        """Upload images to dataset"""
        dataset = self.get_object()
        serializer = BulkImageUploadSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        images = data['images']
        image_type = data['image_type']
        
        uploaded_images = []
        failed_uploads = []
        
        for idx, image_file in enumerate(images):
            try:
                # Create TrainingImage record
                training_image = TrainingImage.objects.create(
                    dataset=dataset,
                    image_file=image_file,
                    image_name=image_file.name,
                    image_type=image_type,
                    width=0,  # Will be updated after processing
                    height=0,
                    file_size_kb=image_file.size // 1024
                )
                
                # Process image to get dimensions
                from PIL import Image
                img = Image.open(image_file)
                training_image.width = img.width
                training_image.height = img.height
                training_image.save()
                
                uploaded_images.append(training_image.id)
                
            except Exception as e:
                failed_uploads.append({
                    'file': image_file.name,
                    'error': str(e)
                })
        
        return Response({
            'status': 'success',
            'uploaded_count': len(uploaded_images),
            'failed_count': len(failed_uploads),
            'uploaded_images': uploaded_images,
            'failed_uploads': failed_uploads
        })


class TrainingJobViewSet(viewsets.ModelViewSet):
    """ViewSet for managing training jobs"""
    
    queryset = TrainingJob.objects.all()
    serializer_class = TrainingJobSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by status if provided
        status_filter = self.request.query_params.get('status', None)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Filter by training type
        training_type = self.request.query_params.get('training_type', None)
        if training_type:
            queryset = queryset.filter(training_type=training_type)
        
        # Filter by user if not admin
        if not self.request.user.is_staff:
            queryset = queryset.filter(created_by=self.request.user)
        
        return queryset.order_by('-created_at')
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @action(detail=False, methods=['post'])
    def create_job(self, request):
        """Create a new training job"""
        serializer = TrainingJobCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        try:
            # Get dataset
            dataset = get_object_or_404(Dataset, id=data['dataset_id'])
            
            if dataset.status != 'ready':
                return Response({
                    'error': 'Dataset is not ready for training'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Create training job
            training_job = TrainingJob.objects.create(
                name=data['name'],
                description=data.get('description', ''),
                training_type=data['training_type'],
                model_architecture=data['model_architecture'],
                dataset=dataset,
                epochs=data['epochs'],
                batch_size=data['batch_size'],
                learning_rate=data['learning_rate'],
                weight_decay=data['weight_decay'],
                momentum=data['momentum'],
                image_size=data['image_size'],
                augmentation=data['augmentation'],
                mixup=data['mixup'],
                mosaic=data['mosaic'],
                use_pretrained=data['use_pretrained'],
                pretrained_model_path=data.get('pretrained_model_path', ''),
                created_by=request.user
            )
            
            # Create training directories
            job_dir = Path(settings.TRAINING_LOGS_PATH) / f"job_{training_job.id}"
            job_dir.mkdir(parents=True, exist_ok=True)
            
            training_job.output_path = str(job_dir)
            training_job.logs_path = str(job_dir / 'logs')
            training_job.model_output_path = str(job_dir / 'models')
            training_job.save()
            
            # Create subdirectories
            (job_dir / 'logs').mkdir(exist_ok=True)
            (job_dir / 'models').mkdir(exist_ok=True)
            (job_dir / 'reports').mkdir(exist_ok=True)
            
            serializer = TrainingJobSerializer(training_job)
            return Response({
                'status': 'success',
                'message': 'Training job created successfully',
                'training_job': serializer.data
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Training job creation failed: {e}")
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        """Start a training job"""
        training_job = self.get_object()
        
        if training_job.status != 'pending':
            return Response({
                'error': f'Job cannot be started. Current status: {training_job.status}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Import training engine
            from .training_engine import DjangoTrainingEngine
            
            # Update job status
            training_job.status = 'running'
            training_job.started_at = timezone.now()
            training_job.save()
            
            # Log start event
            TrainingLog.objects.create(
                training_job=training_job,
                level='INFO',
                message=f'Training job started by {request.user.username}',
                epoch=0
            )
            
            # Start actual YOLO training
            engine = DjangoTrainingEngine()
            
            # Run training in background (for production, use Celery)
            import threading
            training_thread = threading.Thread(
                target=engine.start_training_job,
                args=(training_job.id,)
            )
            training_thread.start()
            
            return Response({
                'status': 'success',
                'message': 'Training job started successfully',
                'job_id': training_job.id,
                'monitor_url': f'/api/ml-training/jobs/{training_job.id}/logs/'
            })
            
        except Exception as e:
            training_job.status = 'failed'
            training_job.save()
            logger.error(f"Training job start failed: {e}")
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def stop(self, request, pk=None):
        """Stop a running training job"""
        training_job = self.get_object()
        
        if training_job.status != 'running':
            return Response({
                'error': f'Job cannot be stopped. Current status: {training_job.status}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            training_job.status = 'cancelled'
            training_job.completed_at = timezone.now()
            training_job.save()
            
            # Log stop event
            TrainingLog.objects.create(
                training_job=training_job,
                level='INFO',
                message=f'Training job stopped by {request.user.username}',
                epoch=training_job.current_epoch
            )
            
            return Response({
                'status': 'success',
                'message': 'Training job stopped successfully'
            })
            
        except Exception as e:
            logger.error(f"Training job stop failed: {e}")
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['put'])
    def update_status(self, request, pk=None):
        """Update training job status (for training engine use)"""
        training_job = self.get_object()
        serializer = TrainingJobStatusSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        try:
            # Update job status
            training_job.status = data['status']
            
            if 'current_epoch' in data:
                training_job.current_epoch = data['current_epoch']
            
            if 'progress_percentage' in data:
                training_job.progress_percentage = data['progress_percentage']
            
            if 'memory_usage_mb' in data:
                training_job.memory_usage_mb = data['memory_usage_mb']
            
            if 'gpu_memory_usage_mb' in data:
                training_job.gpu_memory_usage_mb = data['gpu_memory_usage_mb']
            
            # Set completion time if finished
            if data['status'] in ['completed', 'failed', 'cancelled']:
                training_job.completed_at = timezone.now()
            
            training_job.save()
            
            # Log status update
            if 'message' in data and data['message']:
                TrainingLog.objects.create(
                    training_job=training_job,
                    level='INFO',
                    message=data['message'],
                    epoch=training_job.current_epoch
                )
            
            return Response({
                'status': 'success',
                'message': 'Status updated successfully'
            })
            
        except Exception as e:
            logger.error(f"Status update failed: {e}")
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['get'])
    def metrics(self, request, pk=None):
        """Get training metrics for a job"""
        training_job = self.get_object()
        
        # Get query parameters
        metric_type = request.query_params.get('metric_type', None)
        epoch_from = request.query_params.get('epoch_from', 0)
        epoch_to = request.query_params.get('epoch_to', training_job.current_epoch)
        
        # Build query
        metrics_query = TrainingMetric.objects.filter(
            training_job=training_job,
            epoch__gte=epoch_from,
            epoch__lte=epoch_to
        )
        
        if metric_type:
            metrics_query = metrics_query.filter(metric_type=metric_type)
        
        metrics = metrics_query.order_by('epoch', 'metric_type')
        serializer = TrainingMetricSerializer(metrics, many=True)
        
        return Response({
            'training_job': training_job.name,
            'metrics': serializer.data
        })
    
    @action(detail=True, methods=['get'])
    def logs(self, request, pk=None):
        """Get training logs for a job"""
        training_job = self.get_object()
        
        # Get query parameters
        level = request.query_params.get('level', None)
        epoch = request.query_params.get('epoch', None)
        limit = int(request.query_params.get('limit', 100))
        
        # Build query
        logs_query = TrainingLog.objects.filter(training_job=training_job)
        
        if level:
            logs_query = logs_query.filter(level=level)
        
        if epoch is not None:
            logs_query = logs_query.filter(epoch=epoch)
        
        logs = logs_query.order_by('-timestamp')[:limit]
        serializer = TrainingLogSerializer(logs, many=True)
        
        return Response({
            'training_job': training_job.name,
            'logs': serializer.data
        })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def training_stats(request):
    """Get training system statistics"""
    
    try:
        # Basic job counts
        total_jobs = TrainingJob.objects.count()
        running_jobs = TrainingJob.objects.filter(status='running').count()
        completed_jobs = TrainingJob.objects.filter(status='completed').count()
        failed_jobs = TrainingJob.objects.filter(status='failed').count()
        
        # Recent activity
        now = timezone.now()
        jobs_24h = TrainingJob.objects.filter(created_at__gte=now - timedelta(hours=24)).count()
        jobs_7d = TrainingJob.objects.filter(created_at__gte=now - timedelta(days=7)).count()
        jobs_30d = TrainingJob.objects.filter(created_at__gte=now - timedelta(days=30)).count()
        
        # Performance metrics
        completed_job_stats = TrainingJob.objects.filter(status='completed').aggregate(
            avg_duration=Avg('completed_at') - Avg('started_at'),
            avg_memory=Avg('memory_usage_mb')
        )
        
        # Dataset stats
        total_datasets = Dataset.objects.count()
        total_images = TrainingImage.objects.count()
        
        # Calculate average training time in hours
        avg_training_time_hours = 0
        if completed_jobs > 0:
            total_duration = sum([
                (job.completed_at - job.started_at).total_seconds() 
                for job in TrainingJob.objects.filter(
                    status='completed', 
                    started_at__isnull=False, 
                    completed_at__isnull=False
                )
            ])
            avg_training_time_hours = total_duration / (completed_jobs * 3600)
        
        data = {
            'total_jobs': total_jobs,
            'running_jobs': running_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'jobs_last_24h': jobs_24h,
            'jobs_last_7d': jobs_7d,
            'jobs_last_30d': jobs_30d,
            'avg_training_time_hours': round(avg_training_time_hours, 2),
            'total_gpu_hours': 0.0,  # TODO: Calculate from job durations with GPU usage
            'avg_memory_usage_mb': completed_job_stats['avg_memory'] or 0,
            'avg_final_accuracy': 0.0,  # TODO: Calculate from training reports
            'best_model_accuracy': 0.0,  # TODO: Get from best performing model
            'total_datasets': total_datasets,
            'total_training_images': total_images
        }
        
        serializer = TrainingStatsSerializer(data)
        return Response(serializer.data)
        
    except Exception as e:
        logger.error(f"Training stats calculation failed: {e}")
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Additional ViewSets
class TrainingMetricViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing training metrics"""
    
    queryset = TrainingMetric.objects.all()
    serializer_class = TrainingMetricSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by training job
        job_id = self.request.query_params.get('job_id', None)
        if job_id:
            queryset = queryset.filter(training_job_id=job_id)
        
        # Filter by metric type
        metric_type = self.request.query_params.get('metric_type', None)
        if metric_type:
            queryset = queryset.filter(metric_type=metric_type)
        
        return queryset.order_by('training_job', 'epoch')


class TrainingLogViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing training logs"""
    
    queryset = TrainingLog.objects.all()
    serializer_class = TrainingLogSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by training job
        job_id = self.request.query_params.get('job_id', None)
        if job_id:
            queryset = queryset.filter(training_job_id=job_id)
        
        # Filter by level
        level = self.request.query_params.get('level', None)
        if level:
            queryset = queryset.filter(level=level)
        
        return queryset.order_by('-timestamp')


class ModelCheckpointViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing model checkpoints"""
    
    queryset = ModelCheckpoint.objects.all()
    serializer_class = ModelCheckpointSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by training job
        job_id = self.request.query_params.get('job_id', None)
        if job_id:
            queryset = queryset.filter(training_job_id=job_id)
        
        return queryset.order_by('training_job', '-epoch')