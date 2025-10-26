from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.utils import timezone
from .models import (
    Dataset, TrainingJob, TrainingMetric, TrainingLog, ModelCheckpoint,
    TrainingImage, TrainingReport
)


class TrainingImageInline(admin.TabularInline):
    model = TrainingImage
    extra = 0
    readonly_fields = ['width', 'height', 'file_size_kb', 'is_corrupted', 'uploaded_at']
    fields = [
        'image_name', 'image_type', 'width', 'height', 'file_size_kb',
        'has_annotations', 'annotation_count', 'is_corrupted'
    ]


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'dataset_type', 'status', 'total_images',
        'train_images', 'val_images', 'test_images', 'created_at'
    ]
    list_filter = ['dataset_type', 'status', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = [
        'total_images', 'train_images', 'val_images', 'test_images',
        'class_distribution', 'created_at', 'updated_at'
    ]
    inlines = [TrainingImageInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'dataset_type', 'status')
        }),
        ('Dataset Statistics', {
            'fields': (
                'total_images', 'train_images', 'val_images', 'test_images',
                'class_names', 'class_distribution'
            )
        }),
        ('Dataset Splits', {
            'fields': ('train_split', 'val_split', 'test_split')
        }),
        ('File Paths', {
            'fields': ('dataset_path', 'annotations_path'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def save_model(self, request, obj, form, change):
        if not change:
            obj.created_by = request.user
        super().save_model(request, obj, form, change)
    
    actions = ['mark_ready', 'mark_processing']
    
    def mark_ready(self, request, queryset):
        updated = queryset.update(status='ready')
        self.message_user(request, f'{updated} datasets marked as ready.')
    mark_ready.short_description = "Mark selected datasets as ready"
    
    def mark_processing(self, request, queryset):
        updated = queryset.update(status='processing')
        self.message_user(request, f'{updated} datasets marked as processing.')
    mark_processing.short_description = "Mark selected datasets as processing"


class TrainingMetricInline(admin.TabularInline):
    model = TrainingMetric
    extra = 0
    readonly_fields = ['timestamp']
    fields = ['epoch', 'metric_type', 'metric_name', 'train_value', 'val_value']


class TrainingLogInline(admin.TabularInline):
    model = TrainingLog
    extra = 0
    readonly_fields = ['timestamp']
    fields = ['level', 'message', 'epoch', 'timestamp']


@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'training_type', 'model_architecture', 'status',
        'current_epoch', 'progress_percentage', 'created_at', 'duration_display'
    ]
    list_filter = [
        'training_type', 'model_architecture', 'status', 'gpu_used', 'created_at'
    ]
    search_fields = ['name', 'description', 'job_id']
    readonly_fields = [
        'job_id', 'current_epoch', 'progress_percentage',
        'started_at', 'completed_at', 'estimated_completion',
        'memory_usage_mb', 'gpu_memory_usage_mb',
        'output_path', 'model_output_path', 'logs_path',
        'created_at', 'updated_at', 'duration_display'
    ]
    inlines = [TrainingMetricInline, TrainingLogInline]
    
    fieldsets = (
        ('Job Information', {
            'fields': ('job_id', 'name', 'description', 'status')
        }),
        ('Training Configuration', {
            'fields': (
                'training_type', 'model_architecture', 'dataset',
                'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'momentum'
            )
        }),
        ('Image Processing', {
            'fields': ('image_size', 'augmentation', 'mixup', 'mosaic'),
            'classes': ('collapse',)
        }),
        ('Pre-trained Model', {
            'fields': ('use_pretrained', 'pretrained_model_path'),
            'classes': ('collapse',)
        }),
        ('Progress & Timing', {
            'fields': (
                'current_epoch', 'progress_percentage',
                'started_at', 'completed_at', 'estimated_completion', 'duration_display'
            )
        }),
        ('Resource Usage', {
            'fields': (
                'gpu_used', 'cpu_cores', 'memory_usage_mb', 'gpu_memory_usage_mb'
            ),
            'classes': ('collapse',)
        }),
        ('Output Paths', {
            'fields': ('output_path', 'model_output_path', 'logs_path'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def duration_display(self, obj):
        if obj.duration:
            total_seconds = int(obj.duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours:02d}:{minutes:02d}:{total_seconds % 60:02d}"
        return "N/A"
    duration_display.short_description = "Duration"
    
    def save_model(self, request, obj, form, change):
        if not change:
            obj.created_by = request.user
        super().save_model(request, obj, form, change)
    
    actions = ['cancel_jobs', 'restart_failed_jobs']
    
    def cancel_jobs(self, request, queryset):
        cancelled = 0
        for job in queryset.filter(status__in=['pending', 'running']):
            job.status = 'cancelled'
            job.completed_at = timezone.now()
            job.save()
            cancelled += 1
        
        self.message_user(request, f'{cancelled} jobs cancelled.')
    cancel_jobs.short_description = "Cancel selected jobs"
    
    def restart_failed_jobs(self, request, queryset):
        restarted = 0
        for job in queryset.filter(status='failed'):
            job.status = 'pending'
            job.current_epoch = 0
            job.progress_percentage = 0.0
            job.started_at = None
            job.completed_at = None
            job.save()
            restarted += 1
        
        self.message_user(request, f'{restarted} jobs restarted.')
    restart_failed_jobs.short_description = "Restart selected failed jobs"


@admin.register(TrainingMetric)
class TrainingMetricAdmin(admin.ModelAdmin):
    list_display = [
        'training_job', 'epoch', 'metric_type', 'metric_name',
        'train_value', 'val_value', 'timestamp'
    ]
    list_filter = ['metric_type', 'timestamp', 'training_job__training_type']
    search_fields = ['training_job__name', 'metric_name']
    readonly_fields = ['timestamp']
    
    fieldsets = (
        ('Metric Information', {
            'fields': ('training_job', 'epoch', 'metric_type', 'metric_name')
        }),
        ('Values', {
            'fields': ('train_value', 'val_value')
        }),
        ('Metadata', {
            'fields': ('timestamp',)
        })
    )


@admin.register(TrainingLog)
class TrainingLogAdmin(admin.ModelAdmin):
    list_display = ['training_job', 'level', 'message_preview', 'epoch', 'timestamp']
    list_filter = ['level', 'timestamp', 'training_job__status']
    search_fields = ['training_job__name', 'message']
    readonly_fields = ['timestamp']
    
    fieldsets = (
        ('Log Information', {
            'fields': ('training_job', 'level', 'epoch')
        }),
        ('Message', {
            'fields': ('message',)
        }),
        ('Metadata', {
            'fields': ('timestamp',)
        })
    )
    
    def message_preview(self, obj):
        return obj.message[:100] + "..." if len(obj.message) > 100 else obj.message
    message_preview.short_description = "Message"


@admin.register(ModelCheckpoint)
class ModelCheckpointAdmin(admin.ModelAdmin):
    list_display = [
        'training_job', 'epoch', 'file_size_mb', 'val_loss',
        'val_accuracy', 'is_best', 'is_latest', 'created_at'
    ]
    list_filter = ['is_best', 'is_latest', 'created_at']
    search_fields = ['training_job__name', 'checkpoint_path']
    readonly_fields = ['created_at']
    
    fieldsets = (
        ('Checkpoint Information', {
            'fields': ('training_job', 'epoch', 'checkpoint_file', 'checkpoint_path')
        }),
        ('File Information', {
            'fields': ('file_size_mb',)
        }),
        ('Performance Metrics', {
            'fields': ('val_loss', 'val_accuracy', 'val_map50')
        }),
        ('Status', {
            'fields': ('is_best', 'is_latest')
        }),
        ('Metadata', {
            'fields': ('created_at',)
        })
    )


@admin.register(TrainingImage)
class TrainingImageAdmin(admin.ModelAdmin):
    list_display = [
        'image_name', 'dataset', 'image_type', 'width', 'height',
        'file_size_kb', 'has_annotations', 'is_corrupted', 'uploaded_at'
    ]
    list_filter = [
        'image_type', 'has_annotations', 'is_corrupted', 'dataset', 'uploaded_at'
    ]
    search_fields = ['image_name', 'dataset__name']
    readonly_fields = [
        'width', 'height', 'file_size_kb', 'is_corrupted',
        'has_annotations', 'annotation_count', 'uploaded_at', 'image_preview'
    ]
    
    fieldsets = (
        ('Image Information', {
            'fields': ('dataset', 'image_file', 'image_preview', 'image_name', 'image_type')
        }),
        ('Image Properties', {
            'fields': ('width', 'height', 'file_size_kb', 'is_corrupted')
        }),
        ('Annotations', {
            'fields': (
                'annotations_file', 'has_annotations', 
                'annotation_count', 'annotations_json'
            ),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('uploaded_at',),
            'classes': ('collapse',)
        })
    )
    
    def image_preview(self, obj):
        if obj.image_file:
            return format_html(
                '<img src="{}" style="max-width: 200px; max-height: 200px;" />',
                obj.image_file.url
            )
        return "No image"
    image_preview.short_description = "Image Preview"
    
    actions = ['mark_corrupted', 'mark_not_corrupted']
    
    def mark_corrupted(self, request, queryset):
        updated = queryset.update(is_corrupted=True)
        self.message_user(request, f'{updated} images marked as corrupted.')
    mark_corrupted.short_description = "Mark selected images as corrupted"
    
    def mark_not_corrupted(self, request, queryset):
        updated = queryset.update(is_corrupted=False)
        self.message_user(request, f'{updated} images marked as not corrupted.')
    mark_not_corrupted.short_description = "Mark selected images as not corrupted"


@admin.register(TrainingReport)
class TrainingReportAdmin(admin.ModelAdmin):
    list_display = [
        'training_job', 'final_train_loss', 'final_val_loss',
        'best_val_accuracy', 'epochs_completed', 'early_stopped', 'generated_at'
    ]
    list_filter = ['early_stopped', 'generated_at']
    search_fields = ['training_job__name']
    readonly_fields = ['generated_at']
    
    fieldsets = (
        ('Report Information', {
            'fields': ('training_job', 'generated_at')
        }),
        ('Final Metrics', {
            'fields': (
                'final_train_loss', 'final_val_loss',
                'best_val_accuracy', 'best_val_map50'
            )
        }),
        ('Training Summary', {
            'fields': (
                'total_training_time', 'epochs_completed',
                'early_stopped', 'early_stop_epoch'
            )
        }),
        ('Resource Usage', {
            'fields': (
                'peak_memory_usage_mb', 'peak_gpu_memory_mb', 'avg_epoch_time_seconds'
            ),
            'classes': ('collapse',)
        }),
        ('Generated Files', {
            'fields': (
                'training_curves_image', 'confusion_matrix_image', 'report_pdf'
            ),
            'classes': ('collapse',)
        }),
        ('Additional Data', {
            'fields': ('class_performance', 'training_config'),
            'classes': ('collapse',)
        })
    )