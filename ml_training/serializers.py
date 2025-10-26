from rest_framework import serializers
from .models import (
    Dataset, TrainingJob, TrainingMetric, TrainingLog, ModelCheckpoint,
    TrainingImage, TrainingReport
)
from pest_detection.models import DetectionModel


class DatasetSerializer(serializers.ModelSerializer):
    """Serializer for Dataset model"""
    
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    split_percentages = serializers.SerializerMethodField()
    
    class Meta:
        model = Dataset
        fields = [
            'id', 'name', 'description', 'dataset_type', 'status',
            'total_images', 'train_images', 'val_images', 'test_images',
            'train_split', 'val_split', 'test_split', 'split_percentages',
            'dataset_path', 'annotations_path', 'class_names', 'class_distribution',
            'created_by', 'created_by_username', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'total_images', 'train_images', 'val_images', 'test_images',
            'class_distribution', 'created_at', 'updated_at'
        ]
    
    def get_split_percentages(self, obj):
        return {
            'train': f"{obj.train_split * 100:.1f}%",
            'val': f"{obj.val_split * 100:.1f}%",
            'test': f"{obj.test_split * 100:.1f}%"
        }


class TrainingJobSerializer(serializers.ModelSerializer):
    """Serializer for TrainingJob model"""
    
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    duration_str = serializers.SerializerMethodField()
    current_metrics = serializers.SerializerMethodField()
    
    class Meta:
        model = TrainingJob
        fields = [
            'id', 'job_id', 'name', 'description', 'training_type',
            'model_architecture', 'dataset', 'dataset_name',
            'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'momentum',
            'image_size', 'augmentation', 'mixup', 'mosaic',
            'pretrained_model_path', 'use_pretrained',
            'status', 'current_epoch', 'progress_percentage',
            'started_at', 'completed_at', 'estimated_completion',
            'gpu_used', 'cpu_cores', 'memory_usage_mb', 'gpu_memory_usage_mb',
            'output_path', 'model_output_path', 'logs_path',
            'created_by', 'created_by_username', 'created_at', 'updated_at',
            'duration_str', 'current_metrics'
        ]
        read_only_fields = [
            'job_id', 'current_epoch', 'progress_percentage',
            'started_at', 'completed_at', 'estimated_completion',
            'memory_usage_mb', 'gpu_memory_usage_mb',
            'output_path', 'model_output_path', 'logs_path',
            'created_at', 'updated_at'
        ]
    
    def get_duration_str(self, obj):
        if obj.duration:
            total_seconds = int(obj.duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return None
    
    def get_current_metrics(self, obj):
        """Get latest metrics for current epoch"""
        if obj.current_epoch > 0:
            latest_metrics = TrainingMetric.objects.filter(
                training_job=obj,
                epoch=obj.current_epoch
            ).values('metric_type', 'train_value', 'val_value')
            
            metrics_dict = {}
            for metric in latest_metrics:
                metrics_dict[metric['metric_type']] = {
                    'train': metric['train_value'],
                    'val': metric['val_value']
                }
            return metrics_dict
        return {}


class TrainingMetricSerializer(serializers.ModelSerializer):
    """Serializer for TrainingMetric model"""
    
    training_job_name = serializers.CharField(source='training_job.name', read_only=True)
    
    class Meta:
        model = TrainingMetric
        fields = [
            'id', 'training_job', 'training_job_name', 'epoch',
            'metric_type', 'metric_name', 'train_value', 'val_value', 'timestamp'
        ]
        read_only_fields = ['timestamp']


class TrainingLogSerializer(serializers.ModelSerializer):
    """Serializer for TrainingLog model"""
    
    training_job_name = serializers.CharField(source='training_job.name', read_only=True)
    
    class Meta:
        model = TrainingLog
        fields = [
            'id', 'training_job', 'training_job_name', 'level',
            'message', 'epoch', 'timestamp'
        ]
        read_only_fields = ['timestamp']


class ModelCheckpointSerializer(serializers.ModelSerializer):
    """Serializer for ModelCheckpoint model"""
    
    training_job_name = serializers.CharField(source='training_job.name', read_only=True)
    
    class Meta:
        model = ModelCheckpoint
        fields = [
            'id', 'training_job', 'training_job_name', 'epoch',
            'checkpoint_file', 'checkpoint_path', 'file_size_mb',
            'val_loss', 'val_accuracy', 'val_map50',
            'is_best', 'is_latest', 'created_at'
        ]
        read_only_fields = ['created_at']


class TrainingImageSerializer(serializers.ModelSerializer):
    """Serializer for TrainingImage model"""
    
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = TrainingImage
        fields = [
            'id', 'dataset', 'dataset_name', 'image_file', 'image_name',
            'image_type', 'width', 'height', 'file_size_kb',
            'annotations_file', 'annotations_json', 'is_corrupted',
            'has_annotations', 'annotation_count', 'uploaded_at'
        ]
        read_only_fields = [
            'width', 'height', 'file_size_kb', 'is_corrupted',
            'has_annotations', 'annotation_count', 'uploaded_at'
        ]


class TrainingReportSerializer(serializers.ModelSerializer):
    """Serializer for TrainingReport model"""
    
    training_job_name = serializers.CharField(source='training_job.name', read_only=True)
    
    class Meta:
        model = TrainingReport
        fields = [
            'id', 'training_job', 'training_job_name',
            'final_train_loss', 'final_val_loss', 'best_val_accuracy', 'best_val_map50',
            'total_training_time', 'epochs_completed', 'early_stopped', 'early_stop_epoch',
            'peak_memory_usage_mb', 'peak_gpu_memory_mb', 'avg_epoch_time_seconds',
            'training_curves_image', 'confusion_matrix_image', 'report_pdf',
            'class_performance', 'training_config', 'generated_at'
        ]
        read_only_fields = ['generated_at']


class TrainingJobCreateSerializer(serializers.Serializer):
    """Serializer for creating new training jobs"""
    
    name = serializers.CharField(max_length=200)
    description = serializers.CharField(required=False, allow_blank=True)
    training_type = serializers.ChoiceField(
        choices=['classification', 'object_detection', 'fine_tuning'],
        default='object_detection'
    )
    model_architecture = serializers.ChoiceField(
        choices=[
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'resnet18', 'resnet50', 'custom'
        ],
        default='yolov8n'
    )
    dataset_id = serializers.IntegerField()
    
    # Hyperparameters
    epochs = serializers.IntegerField(min_value=1, max_value=1000, default=50)
    batch_size = serializers.IntegerField(min_value=1, max_value=128, default=8)
    learning_rate = serializers.FloatField(min_value=0.0001, max_value=1.0, default=0.001)
    weight_decay = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.0001)
    momentum = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.9)
    
    # Image preprocessing
    image_size = serializers.IntegerField(min_value=32, max_value=2048, default=640)
    augmentation = serializers.BooleanField(default=True)
    mixup = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.0)
    mosaic = serializers.FloatField(min_value=0.0, max_value=1.0, default=1.0)
    
    # Pre-trained model
    use_pretrained = serializers.BooleanField(default=True)
    pretrained_model_path = serializers.CharField(required=False, allow_blank=True)


class TrainingJobStatusSerializer(serializers.Serializer):
    """Serializer for training job status updates"""
    
    status = serializers.ChoiceField(
        choices=['pending', 'running', 'completed', 'failed', 'cancelled', 'paused']
    )
    current_epoch = serializers.IntegerField(min_value=0, required=False)
    progress_percentage = serializers.FloatField(min_value=0.0, max_value=100.0, required=False)
    
    # Resource usage
    memory_usage_mb = serializers.IntegerField(min_value=0, required=False)
    gpu_memory_usage_mb = serializers.IntegerField(min_value=0, required=False)
    
    # Optional message
    message = serializers.CharField(required=False, allow_blank=True)


class TrainingStatsSerializer(serializers.Serializer):
    """Serializer for training statistics"""
    
    total_jobs = serializers.IntegerField()
    running_jobs = serializers.IntegerField()
    completed_jobs = serializers.IntegerField()
    failed_jobs = serializers.IntegerField()
    
    # Recent activity
    jobs_last_24h = serializers.IntegerField()
    jobs_last_7d = serializers.IntegerField()
    jobs_last_30d = serializers.IntegerField()
    
    # Resource usage
    avg_training_time_hours = serializers.FloatField()
    total_gpu_hours = serializers.FloatField()
    avg_memory_usage_mb = serializers.FloatField()
    
    # Performance metrics
    avg_final_accuracy = serializers.FloatField()
    best_model_accuracy = serializers.FloatField()
    
    # Dataset stats
    total_datasets = serializers.IntegerField()
    total_training_images = serializers.IntegerField()


class BulkImageUploadSerializer(serializers.Serializer):
    """Serializer for bulk image upload to datasets"""
    
    dataset_id = serializers.IntegerField()
    images = serializers.ListField(
        child=serializers.ImageField(),
        min_length=1,
        max_length=100  # Limit bulk upload
    )
    image_type = serializers.ChoiceField(
        choices=['train', 'val', 'test'],
        default='train'
    )
    
    # Optional annotations
    annotations = serializers.ListField(
        child=serializers.FileField(),
        required=False
    )


class ModelExportSerializer(serializers.Serializer):
    """Serializer for exporting trained models"""
    
    training_job_id = serializers.IntegerField()
    export_format = serializers.ChoiceField(
        choices=['pytorch', 'onnx', 'tensorrt', 'openvino'],
        default='pytorch'
    )
    include_metadata = serializers.BooleanField(default=True)
    include_config = serializers.BooleanField(default=True)
