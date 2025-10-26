from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import (
    PestType, DetectionModel, DetectionSession, PestDetection,
    PestCount, DetectionFeedback, DetectionAnalytics
)


@admin.register(PestType)
class PestTypeAdmin(admin.ModelAdmin):
    list_display = ['name', 'scientific_name', 'confidence_threshold', 'created_at']
    list_filter = ['created_at', 'confidence_threshold']
    search_fields = ['name', 'scientific_name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'scientific_name', 'description')
        }),
        ('Damage Information', {
            'fields': ('damage_description', 'diseases_caused'),
            'classes': ('collapse',)
        }),
        ('Management Information', {
            'fields': ('pesticide_recommendations', 'biological_control', 'cultural_control'),
            'classes': ('collapse',)
        }),
        ('Detection Settings', {
            'fields': ('confidence_threshold',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(DetectionModel)
class DetectionModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'version', 'status', 'is_active', 'accuracy', 'created_at']
    list_filter = ['model_type', 'status', 'is_active', 'created_at']
    search_fields = ['name', 'version', 'training_dataset']
    readonly_fields = ['created_at', 'updated_at']
    filter_horizontal = ['pest_types']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'model_type', 'version', 'status', 'is_active')
        }),
        ('Model Files', {
            'fields': ('model_file', 'model_path')
        }),
        ('Training Configuration', {
            'fields': ('training_dataset', 'training_epochs', 'batch_size', 'learning_rate'),
            'classes': ('collapse',)
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score', 'map50', 'map95'),
            'classes': ('collapse',)
        }),
        ('Pest Types', {
            'fields': ('pest_types',)
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def save_model(self, request, obj, form, change):
        if not change:  # If creating new object
            obj.created_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(DetectionSession)
class DetectionSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'user', 'model_used', 'confidence_threshold', 'created_at']
    list_filter = ['created_at', 'model_used']
    search_fields = ['session_id', 'user__username']
    readonly_fields = ['session_id', 'created_at']
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if not request.user.is_superuser:
            qs = qs.filter(user=request.user)
        return qs


class PestCountInline(admin.TabularInline):
    model = PestCount
    extra = 0
    readonly_fields = ['pest_type', 'count', 'confidence_avg']


@admin.register(PestDetection)
class PestDetectionAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'image_name', 'total_pests_detected', 'inference_time_ms', 'processed_at']
    list_filter = ['processed_at', 'session__model_used', 'farm_location']
    search_fields = ['image_name', 'farm_location', 'session__session_id']
    readonly_fields = [
        'total_pests_detected', 'detection_results', 'raw_detections', 
        'inference_time_ms', 'processed_at', 'image_preview'
    ]
    inlines = [PestCountInline]
    
    fieldsets = (
        ('Detection Information', {
            'fields': ('session', 'image', 'image_preview', 'image_name')
        }),
        ('Results', {
            'fields': ('total_pests_detected', 'detection_results', 'inference_time_ms')
        }),
        ('Location', {
            'fields': ('latitude', 'longitude', 'farm_location'),
            'classes': ('collapse',)
        }),
        ('Technical Details', {
            'fields': ('image_size_width', 'image_size_height', 'raw_detections', 'processed_at'),
            'classes': ('collapse',)
        })
    )
    
    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-width: 200px; max-height: 200px;" />',
                obj.image.url
            )
        return "No image"
    image_preview.short_description = "Image Preview"
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if not request.user.is_superuser:
            qs = qs.filter(session__user=request.user)
        return qs


@admin.register(DetectionFeedback)
class DetectionFeedbackAdmin(admin.ModelAdmin):
    list_display = ['detection', 'user', 'feedback_type', 'pest_type', 'corrected_count', 'created_at']
    list_filter = ['feedback_type', 'pest_type', 'created_at']
    search_fields = ['detection__id', 'user__username', 'comments']
    readonly_fields = ['created_at']
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if not request.user.is_superuser:
            qs = qs.filter(user=request.user)
        return qs


@admin.register(DetectionAnalytics)
class DetectionAnalyticsAdmin(admin.ModelAdmin):
    list_display = ['date', 'pest_type', 'total_detections', 'total_count', 'avg_confidence', 'most_affected_location']
    list_filter = ['date', 'pest_type', 'most_affected_location']
    search_fields = ['pest_type__name', 'most_affected_location']
    readonly_fields = ['created_at', 'updated_at']
    date_hierarchy = 'date'
    
    fieldsets = (
        ('Date and Pest', {
            'fields': ('date', 'pest_type')
        }),
        ('Detection Statistics', {
            'fields': ('total_detections', 'total_count', 'avg_confidence')
        }),
        ('Location Statistics', {
            'fields': ('locations_count', 'most_affected_location')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


# Custom admin site configuration
admin.site.site_header = "AgriShield ML Administration"
admin.site.site_title = "AgriShield ML Admin"
admin.site.index_title = "Welcome to AgriShield ML Administration"
