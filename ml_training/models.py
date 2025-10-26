from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid
import os


class Dataset(models.Model):
    """Model for managing training datasets"""
    
    DATASET_TYPES = [
        ('classification', 'Classification'),
        ('object_detection', 'Object Detection'),
        ('mixed', 'Mixed'),
    ]
    
    STATUS_CHOICES = [
        ('preparing', 'Preparing'),
        ('ready', 'Ready'),
        ('processing', 'Processing'),
        ('error', 'Error'),
    ]
    
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    dataset_type = models.CharField(max_length=20, choices=DATASET_TYPES, default='object_detection')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='preparing')
    
    # Dataset statistics
    total_images = models.IntegerField(default=0)
    train_images = models.IntegerField(default=0)
    val_images = models.IntegerField(default=0)
    test_images = models.IntegerField(default=0)
    
    # Dataset splits (percentages)
    train_split = models.FloatField(default=0.7, validators=[MinValueValidator(0.1), MaxValueValidator(0.9)])
    val_split = models.FloatField(default=0.2, validators=[MinValueValidator(0.1), MaxValueValidator(0.9)])
    test_split = models.FloatField(default=0.1, validators=[MinValueValidator(0.0), MaxValueValidator(0.9)])
    
    # File paths
    dataset_path = models.CharField(max_length=500, blank=True)
    annotations_path = models.CharField(max_length=500, blank=True)
    
    # Classes in this dataset
    class_names = models.JSONField(default=list)
    class_distribution = models.JSONField(default=dict, blank=True)  # Store class counts
    
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.dataset_type})"


class TrainingJob(models.Model):
    """Model for managing ML training jobs"""
    
    TRAINING_TYPES = [
        ('classification', 'Classification'),
        ('object_detection', 'Object Detection'),
        ('fine_tuning', 'Fine Tuning'),
    ]
    
    MODEL_ARCHITECTURES = [
        ('yolov8n', 'YOLOv8 Nano'),
        ('yolov8s', 'YOLOv8 Small'),
        ('yolov8m', 'YOLOv8 Medium'),
        ('yolov8l', 'YOLOv8 Large'),
        ('yolov8x', 'YOLOv8 Extra Large'),
        ('resnet18', 'ResNet18'),
        ('resnet50', 'ResNet50'),
        ('custom', 'Custom Architecture'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
        ('paused', 'Paused'),
    ]
    
    # Job identification
    job_id = models.UUIDField(default=uuid.uuid4, unique=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Training configuration
    training_type = models.CharField(max_length=20, choices=TRAINING_TYPES, default='object_detection')
    model_architecture = models.CharField(max_length=20, choices=MODEL_ARCHITECTURES, default='yolov8n')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    
    # Hyperparameters
    epochs = models.IntegerField(default=50, validators=[MinValueValidator(1), MaxValueValidator(1000)])
    batch_size = models.IntegerField(default=8, validators=[MinValueValidator(1), MaxValueValidator(128)])
    learning_rate = models.FloatField(default=0.001, validators=[MinValueValidator(0.0001), MaxValueValidator(1.0)])
    weight_decay = models.FloatField(default=0.0001, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    momentum = models.FloatField(default=0.9, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    
    # Image preprocessing
    image_size = models.IntegerField(default=640, validators=[MinValueValidator(32), MaxValueValidator(2048)])
    augmentation = models.BooleanField(default=True)
    mixup = models.FloatField(default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    mosaic = models.FloatField(default=1.0, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    
    # Pre-trained model
    pretrained_model_path = models.CharField(max_length=500, blank=True)
    use_pretrained = models.BooleanField(default=True)
    
    # Job status and timing
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    current_epoch = models.IntegerField(default=0)
    progress_percentage = models.FloatField(default=0.0)
    
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    estimated_completion = models.DateTimeField(null=True, blank=True)
    
    # Resource utilization
    gpu_used = models.BooleanField(default=False)
    cpu_cores = models.IntegerField(default=1)
    memory_usage_mb = models.IntegerField(default=0)
    gpu_memory_usage_mb = models.IntegerField(default=0)
    
    # Output paths
    output_path = models.CharField(max_length=500, blank=True)
    model_output_path = models.CharField(max_length=500, blank=True)
    logs_path = models.CharField(max_length=500, blank=True)
    
    # User information
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.status})"
    
    @property
    def duration(self):
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class TrainingMetric(models.Model):
    """Model for storing training metrics during training"""
    
    METRIC_TYPES = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1_score', 'F1 Score'),
        ('map50', 'mAP@0.5'),
        ('map95', 'mAP@0.5:0.95'),
        ('lr', 'Learning Rate'),
        ('custom', 'Custom Metric'),
    ]
    
    training_job = models.ForeignKey(TrainingJob, on_delete=models.CASCADE, related_name='metrics')
    epoch = models.IntegerField()
    metric_type = models.CharField(max_length=20, choices=METRIC_TYPES)
    metric_name = models.CharField(max_length=100)  # For custom metrics
    
    # Values for train/validation
    train_value = models.FloatField(null=True, blank=True)
    val_value = models.FloatField(null=True, blank=True)
    
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['training_job', 'epoch', 'metric_type', 'metric_name']
        ordering = ['training_job', 'epoch', 'metric_type']
    
    def __str__(self):
        return f"{self.training_job.name} - Epoch {self.epoch} - {self.metric_type}"


class TrainingLog(models.Model):
    """Model for storing training logs"""
    
    LOG_LEVELS = [
        ('DEBUG', 'Debug'),
        ('INFO', 'Info'),
        ('WARNING', 'Warning'),
        ('ERROR', 'Error'),
        ('CRITICAL', 'Critical'),
    ]
    
    training_job = models.ForeignKey(TrainingJob, on_delete=models.CASCADE, related_name='logs')
    level = models.CharField(max_length=10, choices=LOG_LEVELS, default='INFO')
    message = models.TextField()
    epoch = models.IntegerField(null=True, blank=True)
    
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.training_job.name} - {self.level}: {self.message[:100]}"


class ModelCheckpoint(models.Model):
    """Model for storing model checkpoints during training"""
    
    training_job = models.ForeignKey(TrainingJob, on_delete=models.CASCADE, related_name='checkpoints')
    epoch = models.IntegerField()
    
    # Model file information
    checkpoint_file = models.FileField(upload_to='checkpoints/', blank=True)
    checkpoint_path = models.CharField(max_length=500)
    file_size_mb = models.FloatField(default=0.0)
    
    # Performance at this checkpoint
    val_loss = models.FloatField(null=True, blank=True)
    val_accuracy = models.FloatField(null=True, blank=True)
    val_map50 = models.FloatField(null=True, blank=True)
    
    is_best = models.BooleanField(default=False)
    is_latest = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['training_job', 'epoch']
        ordering = ['-epoch']
    
    def __str__(self):
        return f"{self.training_job.name} - Epoch {self.epoch} Checkpoint"


class TrainingImage(models.Model):
    """Model for managing training images and annotations"""
    
    IMAGE_TYPES = [
        ('train', 'Training'),
        ('val', 'Validation'),
        ('test', 'Test'),
    ]
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='images')
    
    # Image information
    image_file = models.ImageField(upload_to='training_images/')
    image_name = models.CharField(max_length=255)
    image_type = models.CharField(max_length=10, choices=IMAGE_TYPES, default='train')
    
    # Image metadata
    width = models.IntegerField()
    height = models.IntegerField()
    file_size_kb = models.IntegerField(default=0)
    
    # Annotation information
    annotations_file = models.FileField(upload_to='annotations/', blank=True)
    annotations_json = models.JSONField(default=dict, blank=True)  # Store annotations in JSON format
    
    # Quality checks
    is_corrupted = models.BooleanField(default=False)
    has_annotations = models.BooleanField(default=False)
    annotation_count = models.IntegerField(default=0)
    
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['dataset', 'image_name']
    
    def __str__(self):
        return f"{self.dataset.name} - {self.image_name}"


class TrainingReport(models.Model):
    """Model for storing training reports and summaries"""
    
    training_job = models.OneToOneField(TrainingJob, on_delete=models.CASCADE, related_name='report')
    
    # Final metrics
    final_train_loss = models.FloatField(null=True, blank=True)
    final_val_loss = models.FloatField(null=True, blank=True)
    best_val_accuracy = models.FloatField(null=True, blank=True)
    best_val_map50 = models.FloatField(null=True, blank=True)
    
    # Training summary
    total_training_time = models.DurationField(null=True, blank=True)
    epochs_completed = models.IntegerField(default=0)
    early_stopped = models.BooleanField(default=False)
    early_stop_epoch = models.IntegerField(null=True, blank=True)
    
    # Resource usage summary
    peak_memory_usage_mb = models.IntegerField(default=0)
    peak_gpu_memory_mb = models.IntegerField(default=0)
    avg_epoch_time_seconds = models.FloatField(null=True, blank=True)
    
    # Generated files
    training_curves_image = models.ImageField(upload_to='reports/curves/', blank=True)
    confusion_matrix_image = models.ImageField(upload_to='reports/matrices/', blank=True)
    report_pdf = models.FileField(upload_to='reports/pdf/', blank=True)
    
    # Additional metrics
    class_performance = models.JSONField(default=dict, blank=True)  # Per-class metrics
    training_config = models.JSONField(default=dict, blank=True)  # Training configuration used
    
    generated_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Report for {self.training_job.name}"