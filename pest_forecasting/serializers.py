from rest_framework import serializers
from .models import (
    WeatherData, PestForecastModel, PestForecast, ForecastAlert,
    ForecastFeedback, ForecastingAnalytics, HistoricalPestData
)
from pest_detection.models import PestType
from pest_detection.serializers import PestTypeSerializer


class WeatherDataSerializer(serializers.ModelSerializer):
    """Serializer for WeatherData model"""
    
    class Meta:
        model = WeatherData
        fields = [
            'id', 'location_name', 'latitude', 'longitude', 'recorded_at',
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
            'rainfall_1h', 'rainfall_24h', 'cloudiness', 'dew_point',
            'visibility', 'uv_index', 'weather_main', 'weather_description',
            'data_source', 'is_forecast', 'created_at'
        ]
        read_only_fields = ['created_at']


class PestForecastModelSerializer(serializers.ModelSerializer):
    """Serializer for PestForecastModel model"""
    
    pest_type_name = serializers.CharField(source='pest_type.name', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = PestForecastModel
        fields = [
            'id', 'name', 'description', 'model_type', 'status',
            'pest_type', 'pest_type_name', 'model_file', 'model_path',
            'accuracy', 'rmse', 'mae', 'r2_score', 'training_data_points',
            'features_used', 'training_period_start', 'training_period_end',
            'version', 'is_active', 'created_by', 'created_by_username',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class PestForecastSerializer(serializers.ModelSerializer):
    """Serializer for PestForecast model"""
    
    pest_type_name = serializers.CharField(source='pest_type.name', read_only=True)
    model_name = serializers.CharField(source='model_used.name', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = PestForecast
        fields = [
            'id', 'forecast_id', 'pest_type', 'pest_type_name',
            'model_used', 'model_name', 'location_name', 'latitude', 'longitude',
            'forecast_date', 'target_date', 'forecast_type',
            'risk_level', 'risk_score', 'confidence',
            'predicted_pest_count', 'predicted_outbreak_probability',
            'weather_conditions', 'management_recommendations',
            'monitoring_recommendations', 'treatment_recommendations',
            'actual_outcome', 'actual_pest_count', 'forecast_accuracy',
            'created_by', 'created_by_username', 'created_at', 'updated_at'
        ]
        read_only_fields = ['forecast_id', 'created_at', 'updated_at']


class ForecastAlertSerializer(serializers.ModelSerializer):
    """Serializer for ForecastAlert model"""
    
    forecast_location = serializers.CharField(source='forecast.location_name', read_only=True)
    acknowledged_by_username = serializers.CharField(source='acknowledged_by.username', read_only=True)
    
    class Meta:
        model = ForecastAlert
        fields = [
            'id', 'alert_id', 'alert_type', 'status', 'forecast',
            'forecast_location', 'title', 'message', 'priority',
            'valid_from', 'valid_until', 'target_users', 'target_locations',
            'acknowledged_by', 'acknowledged_by_username', 'acknowledged_at',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['alert_id', 'created_at', 'updated_at']


class ForecastFeedbackSerializer(serializers.ModelSerializer):
    """Serializer for ForecastFeedback model"""
    
    user_username = serializers.CharField(source='user.username', read_only=True)
    forecast_location = serializers.CharField(source='forecast.location_name', read_only=True)
    
    class Meta:
        model = ForecastFeedback
        fields = [
            'id', 'forecast', 'forecast_location', 'user', 'user_username',
            'feedback_type', 'observed_pest_count', 'observed_weather_conditions',
            'accuracy_rating', 'comments', 'latitude', 'longitude', 'created_at'
        ]
        read_only_fields = ['created_at']


class ForecastingAnalyticsSerializer(serializers.ModelSerializer):
    """Serializer for ForecastingAnalytics model"""
    
    pest_type_name = serializers.CharField(source='pest_type.name', read_only=True)
    
    class Meta:
        model = ForecastingAnalytics
        fields = [
            'id', 'date', 'pest_type', 'pest_type_name', 'location_name',
            'total_forecasts', 'high_risk_forecasts', 'critical_risk_forecasts',
            'avg_forecast_accuracy', 'total_feedback_count',
            'alerts_generated', 'alerts_acknowledged',
            'avg_temperature', 'avg_humidity', 'total_rainfall',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class HistoricalPestDataSerializer(serializers.ModelSerializer):
    """Serializer for HistoricalPestData model"""
    
    pest_type_name = serializers.CharField(source='pest_type.name', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = HistoricalPestData
        fields = [
            'id', 'pest_type', 'pest_type_name', 'location_name',
            'latitude', 'longitude', 'recorded_date', 'pest_count',
            'infestation_level', 'temperature', 'humidity',
            'rainfall_7days', 'rainfall_14days', 'data_source',
            'verified', 'notes', 'created_by', 'created_by_username', 'created_at'
        ]
        read_only_fields = ['created_at']


class ForecastRequestSerializer(serializers.Serializer):
    """Serializer for forecast generation requests"""
    
    pest_type_id = serializers.IntegerField()
    location_name = serializers.CharField(max_length=200)
    latitude = serializers.DecimalField(max_digits=10, decimal_places=8)
    longitude = serializers.DecimalField(max_digits=11, decimal_places=8)
    
    # Weather data for forecasting
    temperature = serializers.FloatField()
    humidity = serializers.FloatField(min_value=0, max_value=100)
    pressure = serializers.FloatField()
    wind_speed = serializers.FloatField(min_value=0)
    rainfall_1h = serializers.FloatField(min_value=0, default=0.0)
    rainfall_24h = serializers.FloatField(min_value=0, default=0.0)
    cloudiness = serializers.FloatField(min_value=0, max_value=100)
    
    # Forecast parameters
    forecast_days = serializers.IntegerField(min_value=1, max_value=30, default=7)
    include_recommendations = serializers.BooleanField(default=True)


class QuickForecastSerializer(serializers.Serializer):
    """Serializer for quick forecast requests"""
    
    location_name = serializers.CharField(max_length=200)
    latitude = serializers.DecimalField(max_digits=10, decimal_places=8)
    longitude = serializers.DecimalField(max_digits=11, decimal_places=8)
    
    # Optional: specific pest types to forecast
    pest_type_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False
    )
    
    # Optional: use current weather from API
    use_current_weather = serializers.BooleanField(default=True)
    
    # Manual weather data (if not using current weather)
    temperature = serializers.FloatField(required=False)
    humidity = serializers.FloatField(min_value=0, max_value=100, required=False)
    rainfall_24h = serializers.FloatField(min_value=0, required=False)


class ForecastResultSerializer(serializers.Serializer):
    """Serializer for forecast API responses"""
    
    status = serializers.CharField()
    forecast_id = serializers.UUIDField()
    location_name = serializers.CharField()
    
    # Forecast results
    pest_forecasts = serializers.ListField()
    risk_summary = serializers.DictField()
    
    # Weather conditions used
    weather_conditions = serializers.DictField()
    
    # Recommendations
    management_recommendations = serializers.ListField()
    monitoring_recommendations = serializers.ListField()
    treatment_recommendations = serializers.ListField()
    
    # Metadata
    model_used = serializers.CharField()
    forecast_date = serializers.DateField()
    confidence = serializers.FloatField()


class WeatherForecastSerializer(serializers.Serializer):
    """Serializer for weather forecast requests"""
    
    location_name = serializers.CharField(max_length=200)
    latitude = serializers.DecimalField(max_digits=10, decimal_places=8)
    longitude = serializers.DecimalField(max_digits=11, decimal_places=8)
    days = serializers.IntegerField(min_value=1, max_value=7, default=5)


class AlertConfigSerializer(serializers.Serializer):
    """Serializer for alert configuration"""
    
    pest_type_ids = serializers.ListField(child=serializers.IntegerField())
    locations = serializers.ListField(child=serializers.CharField())
    risk_threshold = serializers.ChoiceField(
        choices=['moderate', 'high', 'critical'],
        default='high'
    )
    notification_methods = serializers.ListField(
        child=serializers.ChoiceField(choices=['email', 'sms', 'push']),
        default=['email']
    )
    active = serializers.BooleanField(default=True)
