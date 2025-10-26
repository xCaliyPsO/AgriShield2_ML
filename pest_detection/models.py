from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid
import os


class PestType(models.Model):
    """Model for different types of pests that can be detected"""
    name = models.CharField(max_length=100, unique=True)
    scientific_name = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    
    # Damage information
    damage_description = models.TextField(blank=True)
    diseases_caused = models.JSONField(default=list, blank=True)
    
    # Pest management information
    pesticide_recommendations = models.TextField(blank=True)
    biological_control = models.TextField(blank=True)
    cultural_control = models.TextField(blank=True)
    
    # Thresholds for detection
    confidence_threshold = models.FloatField(
        default=0.25,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name


class DetectionModel(models.Model):
    """Model for storing information about trained ML models"""
    
    MODEL_TYPES = [
        ('yolo', 'YOLO'),
        ('resnet', 'ResNet'),
        ('custom', 'Custom'),
    ]
    
    STATUS_CHOICES = [
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('active', 'Active'),
        ('archived', 'Archived'),
    ]
    
    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES, default='yolo')
    version = models.CharField(max_length=50)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training')
    
    # Model file information
    model_file = models.FileField(upload_to='models/pest_detection/', blank=True)
    model_path = models.CharField(max_length=500, blank=True)  # For existing model files
    
    # Training metadata
    training_dataset = models.CharField(max_length=200, blank=True)
    training_epochs = models.IntegerField(default=50)
    batch_size = models.IntegerField(default=8)
    learning_rate = models.FloatField(default=0.001)
    
    # Performance metrics
    accuracy = models.FloatField(blank=True, null=True)
    precision = models.FloatField(blank=True, null=True)
    recall = models.FloatField(blank=True, null=True)
    f1_score = models.FloatField(blank=True, null=True)
    map50 = models.FloatField(blank=True, null=True)  # mAP@0.5 for YOLO
    map95 = models.FloatField(blank=True, null=True)  # mAP@0.5:0.95 for YOLO
    
    # Classes this model can detect
    pest_types = models.ManyToManyField(PestType, blank=True)
    
    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=False)  # Only one model should be active at a time
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} v{self.version}"
    
    def save(self, *args, **kwargs):
        # Ensure only one model is active at a time
        if self.is_active:
            DetectionModel.objects.filter(is_active=True).update(is_active=False)
        super().save(*args, **kwargs)


class DetectionSession(models.Model):
    """Model for tracking detection sessions"""
    
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    model_used = models.ForeignKey(DetectionModel, on_delete=models.SET_NULL, null=True)
    
    # Detection parameters
    confidence_threshold = models.FloatField(default=0.25)
    iou_threshold = models.FloatField(default=0.50)
    image_size = models.IntegerField(default=512)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Session {self.session_id}"


class PestDetection(models.Model):
    """Model for storing individual pest detection results"""
    
    session = models.ForeignKey(DetectionSession, on_delete=models.CASCADE, related_name='detections')
    
    # Image information
    image = models.ImageField(upload_to='detections/')
    image_name = models.CharField(max_length=255)
    image_size_width = models.IntegerField()
    image_size_height = models.IntegerField()
    
    # Detection results
    total_pests_detected = models.IntegerField(default=0)
    detection_results = models.JSONField(default=dict)  # Store per-class counts
    raw_detections = models.JSONField(default=list, blank=True)  # Store raw detection data
    
    # Processing information
    inference_time_ms = models.FloatField()
    processed_at = models.DateTimeField(auto_now_add=True)
    
    # Location information (if available)
    latitude = models.DecimalField(max_digits=10, decimal_places=8, null=True, blank=True)
    longitude = models.DecimalField(max_digits=11, decimal_places=8, null=True, blank=True)
    farm_location = models.CharField(max_length=255, blank=True)
    
    class Meta:
        ordering = ['-processed_at']
    
    def __str__(self):
        return f"Detection {self.id} - {self.total_pests_detected} pests"


class PestCount(models.Model):
    """Model for storing individual pest counts per detection"""
    
    detection = models.ForeignKey(PestDetection, on_delete=models.CASCADE, related_name='pest_counts')
    pest_type = models.ForeignKey(PestType, on_delete=models.CASCADE)
    count = models.IntegerField(default=0)
    confidence_avg = models.FloatField(blank=True, null=True)  # Average confidence for this pest type
    
    class Meta:
        unique_together = ['detection', 'pest_type']
        ordering = ['pest_type__name']
    
    def __str__(self):
        return f"{self.pest_type.name}: {self.count}"


class DetectionFeedback(models.Model):
    """Model for storing user feedback on detection results"""
    
    FEEDBACK_TYPES = [
        ('correct', 'Correct Detection'),
        ('incorrect', 'Incorrect Detection'),
        ('missed', 'Missed Detection'),
        ('false_positive', 'False Positive'),
    ]
    
    detection = models.ForeignKey(PestDetection, on_delete=models.CASCADE, related_name='feedback')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPES)
    pest_type = models.ForeignKey(PestType, on_delete=models.CASCADE)
    
    # Corrected information
    corrected_count = models.IntegerField(null=True, blank=True)
    comments = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback for Detection {self.detection.id}: {self.feedback_type}"


class DetectionAnalytics(models.Model):
    """Model for storing aggregated analytics data"""
    
    date = models.DateField()
    pest_type = models.ForeignKey(PestType, on_delete=models.CASCADE)
    
    # Daily aggregates
    total_detections = models.IntegerField(default=0)
    total_count = models.IntegerField(default=0)
    avg_confidence = models.FloatField(blank=True, null=True)
    
    # Geographic aggregates (if location data available)
    locations_count = models.IntegerField(default=0)
    most_affected_location = models.CharField(max_length=255, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['date', 'pest_type']
        ordering = ['-date', 'pest_type__name']
    
    def __str__(self):
        return f"{self.pest_type.name} - {self.date}: {self.total_count} pests"
