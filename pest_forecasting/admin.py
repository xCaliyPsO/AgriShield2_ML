from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import (
    WeatherData, PestForecastModel, PestForecast, ForecastAlert,
    ForecastFeedback, ForecastingAnalytics, HistoricalPestData
)


@admin.register(WeatherData)
class WeatherDataAdmin(admin.ModelAdmin):
    list_display = [
        'location_name', 'recorded_at', 'temperature', 'humidity', 
        'rainfall_24h', 'wind_speed', 'is_forecast', 'data_source'
    ]
    list_filter = ['is_forecast', 'data_source', 'weather_main', 'recorded_at']
    search_fields = ['location_name', 'weather_description']
    readonly_fields = ['created_at']
    date_hierarchy = 'recorded_at'
    
    fieldsets = (
        ('Location & Time', {
            'fields': ('location_name', 'latitude', 'longitude', 'recorded_at')
        }),
        ('Basic Weather', {
            'fields': ('temperature', 'humidity', 'pressure', 'cloudiness')
        }),
        ('Wind & Precipitation', {
            'fields': ('wind_speed', 'wind_direction', 'rainfall_1h', 'rainfall_24h')
        }),
        ('Additional Data', {
            'fields': ('dew_point', 'visibility', 'uv_index'),
            'classes': ('collapse',)
        }),
        ('Weather Conditions', {
            'fields': ('weather_main', 'weather_description')
        }),
        ('Metadata', {
            'fields': ('data_source', 'is_forecast', 'created_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(PestForecastModel)
class PestForecastModelAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'pest_type', 'model_type', 'status', 'is_active', 
        'accuracy', 'rmse', 'training_data_points', 'created_at'
    ]
    list_filter = ['model_type', 'status', 'is_active', 'pest_type', 'created_at']
    search_fields = ['name', 'description', 'pest_type__name']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'model_type', 'status', 'pest_type')
        }),
        ('Model Files', {
            'fields': ('model_file', 'model_path', 'is_active')
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'rmse', 'mae', 'r2_score')
        }),
        ('Training Information', {
            'fields': (
                'training_data_points', 'features_used', 
                'training_period_start', 'training_period_end'
            ),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('version', 'created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def save_model(self, request, obj, form, change):
        if not change:
            obj.created_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(PestForecast)
class PestForecastAdmin(admin.ModelAdmin):
    list_display = [
        'forecast_id', 'pest_type', 'location_name', 'target_date', 
        'risk_level', 'risk_score', 'confidence', 'predicted_pest_count'
    ]
    list_filter = [
        'risk_level', 'forecast_type', 'pest_type', 
        'target_date', 'created_at'
    ]
    search_fields = [
        'location_name', 'forecast_id', 'pest_type__name',
        'management_recommendations'
    ]
    readonly_fields = ['forecast_id', 'created_at', 'updated_at']
    date_hierarchy = 'target_date'
    
    fieldsets = (
        ('Forecast Identity', {
            'fields': ('forecast_id', 'pest_type', 'model_used', 'forecast_type')
        }),
        ('Location & Timing', {
            'fields': (
                'location_name', 'latitude', 'longitude', 
                'forecast_date', 'target_date'
            )
        }),
        ('Forecast Results', {
            'fields': (
                'risk_level', 'risk_score', 'confidence',
                'predicted_pest_count', 'predicted_outbreak_probability'
            )
        }),
        ('Weather Conditions', {
            'fields': ('weather_conditions',),
            'classes': ('collapse',)
        }),
        ('Recommendations', {
            'fields': (
                'management_recommendations', 
                'monitoring_recommendations',
                'treatment_recommendations'
            ),
            'classes': ('collapse',)
        }),
        ('Validation', {
            'fields': (
                'actual_outcome', 'actual_pest_count', 'forecast_accuracy'
            ),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if not request.user.is_superuser:
            qs = qs.filter(created_by=request.user)
        return qs


@admin.register(ForecastAlert)
class ForecastAlertAdmin(admin.ModelAdmin):
    list_display = [
        'alert_id', 'alert_type', 'title', 'priority', 'status',
        'valid_from', 'valid_until', 'acknowledged_by'
    ]
    list_filter = [
        'alert_type', 'status', 'priority', 'valid_from', 'created_at'
    ]
    search_fields = ['title', 'message', 'alert_id']
    readonly_fields = ['alert_id', 'created_at', 'updated_at']
    filter_horizontal = ['target_users']
    
    fieldsets = (
        ('Alert Information', {
            'fields': ('alert_id', 'alert_type', 'status', 'forecast')
        }),
        ('Alert Content', {
            'fields': ('title', 'message', 'priority')
        }),
        ('Timing', {
            'fields': ('valid_from', 'valid_until')
        }),
        ('Targeting', {
            'fields': ('target_users', 'target_locations')
        }),
        ('Response', {
            'fields': ('acknowledged_by', 'acknowledged_at'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    actions = ['mark_acknowledged', 'mark_resolved']
    
    def mark_acknowledged(self, request, queryset):
        updated = queryset.update(
            status='acknowledged',
            acknowledged_by=request.user,
            acknowledged_at=timezone.now()
        )
        self.message_user(request, f'{updated} alerts marked as acknowledged.')
    mark_acknowledged.short_description = "Mark selected alerts as acknowledged"
    
    def mark_resolved(self, request, queryset):
        updated = queryset.update(status='resolved')
        self.message_user(request, f'{updated} alerts marked as resolved.')
    mark_resolved.short_description = "Mark selected alerts as resolved"


@admin.register(ForecastFeedback)
class ForecastFeedbackAdmin(admin.ModelAdmin):
    list_display = [
        'forecast', 'user', 'feedback_type', 'accuracy_rating',
        'observed_pest_count', 'created_at'
    ]
    list_filter = ['feedback_type', 'accuracy_rating', 'created_at']
    search_fields = ['forecast__location_name', 'user__username', 'comments']
    readonly_fields = ['created_at']
    
    fieldsets = (
        ('Feedback Information', {
            'fields': ('forecast', 'user', 'feedback_type')
        }),
        ('Observed Data', {
            'fields': (
                'observed_pest_count', 'observed_weather_conditions',
                'latitude', 'longitude'
            )
        }),
        ('Rating & Comments', {
            'fields': ('accuracy_rating', 'comments')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if not request.user.is_superuser:
            qs = qs.filter(user=request.user)
        return qs


@admin.register(ForecastingAnalytics)
class ForecastingAnalyticsAdmin(admin.ModelAdmin):
    list_display = [
        'date', 'pest_type', 'location_name', 'total_forecasts',
        'high_risk_forecasts', 'avg_forecast_accuracy', 'alerts_generated'
    ]
    list_filter = ['date', 'pest_type', 'location_name']
    search_fields = ['pest_type__name', 'location_name']
    readonly_fields = ['created_at', 'updated_at']
    date_hierarchy = 'date'
    
    fieldsets = (
        ('Date & Location', {
            'fields': ('date', 'pest_type', 'location_name')
        }),
        ('Forecast Statistics', {
            'fields': (
                'total_forecasts', 'high_risk_forecasts', 
                'critical_risk_forecasts', 'avg_forecast_accuracy'
            )
        }),
        ('Alert Statistics', {
            'fields': ('alerts_generated', 'alerts_acknowledged')
        }),
        ('Weather Summary', {
            'fields': ('avg_temperature', 'avg_humidity', 'total_rainfall'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(HistoricalPestData)
class HistoricalPestDataAdmin(admin.ModelAdmin):
    list_display = [
        'pest_type', 'location_name', 'recorded_date', 'pest_count',
        'infestation_level', 'temperature', 'humidity', 'verified'
    ]
    list_filter = [
        'pest_type', 'infestation_level', 'verified', 
        'data_source', 'recorded_date'
    ]
    search_fields = ['location_name', 'pest_type__name', 'notes']
    readonly_fields = ['created_at']
    date_hierarchy = 'recorded_date'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('pest_type', 'location_name', 'recorded_date')
        }),
        ('Location', {
            'fields': ('latitude', 'longitude')
        }),
        ('Pest Data', {
            'fields': ('pest_count', 'infestation_level')
        }),
        ('Weather Conditions', {
            'fields': ('temperature', 'humidity', 'rainfall_7days', 'rainfall_14days')
        }),
        ('Data Quality', {
            'fields': ('data_source', 'verified', 'notes')
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at'),
            'classes': ('collapse',)
        })
    )
    
    def save_model(self, request, obj, form, change):
        if not change:
            obj.created_by = request.user
        super().save_model(request, obj, form, change)
    
    actions = ['mark_verified', 'mark_unverified']
    
    def mark_verified(self, request, queryset):
        updated = queryset.update(verified=True)
        self.message_user(request, f'{updated} records marked as verified.')
    mark_verified.short_description = "Mark selected records as verified"
    
    def mark_unverified(self, request, queryset):
        updated = queryset.update(verified=False)
        self.message_user(request, f'{updated} records marked as unverified.')
    mark_unverified.short_description = "Mark selected records as unverified"