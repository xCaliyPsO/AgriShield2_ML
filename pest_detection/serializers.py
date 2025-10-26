from rest_framework import serializers
from .models import (
    PestType, DetectionModel, DetectionSession, PestDetection, 
    PestCount, DetectionFeedback, DetectionAnalytics
)
from django.contrib.auth.models import User


class PestTypeSerializer(serializers.ModelSerializer):
    """Serializer for PestType model"""
    
    class Meta:
        model = PestType
        fields = [
            'id', 'name', 'scientific_name', 'description', 
            'damage_description', 'diseases_caused', 
            'pesticide_recommendations', 'biological_control', 
            'cultural_control', 'confidence_threshold', 
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class DetectionModelSerializer(serializers.ModelSerializer):
    """Serializer for DetectionModel model"""
    
    pest_types = PestTypeSerializer(many=True, read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = DetectionModel
        fields = [
            'id', 'name', 'model_type', 'version', 'status',
            'model_file', 'model_path', 'training_dataset',
            'training_epochs', 'batch_size', 'learning_rate',
            'accuracy', 'precision', 'recall', 'f1_score', 'map50', 'map95',
            'pest_types', 'created_by', 'created_by_username',
            'created_at', 'updated_at', 'is_active'
        ]
        read_only_fields = [
            'created_at', 'updated_at', 'accuracy', 'precision', 
            'recall', 'f1_score', 'map50', 'map95'
        ]


class DetectionSessionSerializer(serializers.ModelSerializer):
    """Serializer for DetectionSession model"""
    
    user_username = serializers.CharField(source='user.username', read_only=True)
    model_name = serializers.CharField(source='model_used.name', read_only=True)
    
    class Meta:
        model = DetectionSession
        fields = [
            'id', 'session_id', 'user', 'user_username',
            'model_used', 'model_name', 'confidence_threshold',
            'iou_threshold', 'image_size', 'created_at'
        ]
        read_only_fields = ['session_id', 'created_at']


class PestCountSerializer(serializers.ModelSerializer):
    """Serializer for PestCount model"""
    
    pest_type_name = serializers.CharField(source='pest_type.name', read_only=True)
    
    class Meta:
        model = PestCount
        fields = [
            'id', 'pest_type', 'pest_type_name',
            'count', 'confidence_avg'
        ]


class PestDetectionSerializer(serializers.ModelSerializer):
    """Serializer for PestDetection model"""
    
    pest_counts = PestCountSerializer(many=True, read_only=True)
    session_id = serializers.UUIDField(source='session.session_id', read_only=True)
    
    class Meta:
        model = PestDetection
        fields = [
            'id', 'session', 'session_id', 'image', 'image_name',
            'image_size_width', 'image_size_height', 'total_pests_detected',
            'detection_results', 'raw_detections', 'inference_time_ms',
            'processed_at', 'latitude', 'longitude', 'farm_location',
            'pest_counts'
        ]
        read_only_fields = [
            'total_pests_detected', 'detection_results', 'raw_detections',
            'inference_time_ms', 'processed_at', 'pest_counts'
        ]


class DetectionFeedbackSerializer(serializers.ModelSerializer):
    """Serializer for DetectionFeedback model"""
    
    user_username = serializers.CharField(source='user.username', read_only=True)
    pest_type_name = serializers.CharField(source='pest_type.name', read_only=True)
    
    class Meta:
        model = DetectionFeedback
        fields = [
            'id', 'detection', 'user', 'user_username',
            'feedback_type', 'pest_type', 'pest_type_name',
            'corrected_count', 'comments', 'created_at'
        ]
        read_only_fields = ['created_at']


class DetectionAnalyticsSerializer(serializers.ModelSerializer):
    """Serializer for DetectionAnalytics model"""
    
    pest_type_name = serializers.CharField(source='pest_type.name', read_only=True)
    
    class Meta:
        model = DetectionAnalytics
        fields = [
            'id', 'date', 'pest_type', 'pest_type_name',
            'total_detections', 'total_count', 'avg_confidence',
            'locations_count', 'most_affected_location',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class PestDetectionCreateSerializer(serializers.Serializer):
    """Serializer for creating new pest detections via API"""
    
    image = serializers.ImageField()
    confidence_threshold = serializers.FloatField(default=0.25, min_value=0.0, max_value=1.0)
    iou_threshold = serializers.FloatField(default=0.50, min_value=0.0, max_value=1.0)
    image_size = serializers.IntegerField(default=512, min_value=32, max_value=2048)
    
    # Optional location data
    latitude = serializers.DecimalField(max_digits=10, decimal_places=8, required=False)
    longitude = serializers.DecimalField(max_digits=11, decimal_places=8, required=False)
    farm_location = serializers.CharField(max_length=255, required=False)
    
    # Optional debug mode
    debug = serializers.BooleanField(default=False)
    disable_black_bug = serializers.BooleanField(default=False)


class PestDetectionResultSerializer(serializers.Serializer):
    """Serializer for pest detection API response"""
    
    status = serializers.CharField()
    session_id = serializers.UUIDField()
    detection_id = serializers.IntegerField()
    
    # Detection results
    pest_counts = serializers.DictField()
    diseases = serializers.DictField()
    recommendations = serializers.DictField()
    
    # Processing info
    inference_time_ms = serializers.FloatField()
    model_used = serializers.CharField()
    
    # Optional debug data
    detections = serializers.ListField(required=False)
    raw_results = serializers.DictField(required=False)


class HealthCheckSerializer(serializers.Serializer):
    """Serializer for health check endpoint"""
    
    status = serializers.CharField()
    model = serializers.CharField()
    classes = serializers.ListField()
    database_status = serializers.CharField()
    model_loaded = serializers.BooleanField()
    total_detections = serializers.IntegerField()
    active_sessions = serializers.IntegerField()


class DetectionStatsSerializer(serializers.Serializer):
    """Serializer for detection statistics"""
    
    total_detections = serializers.IntegerField()
    total_sessions = serializers.IntegerField()
    total_pests_detected = serializers.IntegerField()
    
    # Per pest type stats
    pest_type_stats = serializers.DictField()
    
    # Recent activity
    detections_last_24h = serializers.IntegerField()
    detections_last_7d = serializers.IntegerField()
    detections_last_30d = serializers.IntegerField()
    
    # Model performance
    avg_inference_time_ms = serializers.FloatField()
    avg_confidence = serializers.FloatField()
    
    # Location stats
    unique_locations = serializers.IntegerField()
    most_active_location = serializers.CharField()


class BulkDetectionSerializer(serializers.Serializer):
    """Serializer for bulk image detection"""
    
    images = serializers.ListField(
        child=serializers.ImageField(),
        min_length=1,
        max_length=50  # Limit bulk processing
    )
    confidence_threshold = serializers.FloatField(default=0.25, min_value=0.0, max_value=1.0)
    iou_threshold = serializers.FloatField(default=0.50, min_value=0.0, max_value=1.0)
    image_size = serializers.IntegerField(default=512, min_value=32, max_value=2048)
    
    # Optional metadata for all images
    farm_location = serializers.CharField(max_length=255, required=False)
    batch_name = serializers.CharField(max_length=200, required=False)


class DetectionExportSerializer(serializers.Serializer):
    """Serializer for exporting detection data"""
    
    date_from = serializers.DateField(required=False)
    date_to = serializers.DateField(required=False)
    pest_types = serializers.ListField(
        child=serializers.IntegerField(),
        required=False
    )
    locations = serializers.ListField(
        child=serializers.CharField(),
        required=False
    )
    format = serializers.ChoiceField(
        choices=['csv', 'json', 'excel'],
        default='csv'
    )
    include_images = serializers.BooleanField(default=False)
