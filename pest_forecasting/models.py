from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from pest_detection.models import PestType
import uuid


class WeatherData(models.Model):
    """Model for storing weather data used in forecasting"""
    
    # Location information
    location_name = models.CharField(max_length=200)
    latitude = models.DecimalField(max_digits=10, decimal_places=8)
    longitude = models.DecimalField(max_digits=11, decimal_places=8)
    
    # Weather timestamp
    recorded_at = models.DateTimeField()
    
    # Weather parameters
    temperature = models.FloatField()  # Celsius
    humidity = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])  # Percentage
    pressure = models.FloatField()  # hPa
    wind_speed = models.FloatField(validators=[MinValueValidator(0)])  # km/h
    wind_direction = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(360)])  # degrees
    rainfall_1h = models.FloatField(default=0.0, validators=[MinValueValidator(0)])  # mm
    rainfall_24h = models.FloatField(default=0.0, validators=[MinValueValidator(0)])  # mm
    cloudiness = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])  # Percentage
    
    # Additional weather parameters
    dew_point = models.FloatField(null=True, blank=True)
    visibility = models.FloatField(null=True, blank=True)  # km
    uv_index = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(15)])
    
    # Weather conditions
    weather_main = models.CharField(max_length=50, blank=True)  # Clear, Clouds, Rain, etc.
    weather_description = models.CharField(max_length=100, blank=True)  # More detailed description
    
    # Data source information
    data_source = models.CharField(max_length=100, default='OpenWeatherMap')
    is_forecast = models.BooleanField(default=False)  # True if this is forecast data, False if actual
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['location_name', 'recorded_at']
        ordering = ['-recorded_at']
    
    def __str__(self):
        return f"{self.location_name} - {self.recorded_at} - {self.temperature}°C"


class PestForecastModel(models.Model):
    """Model for managing pest forecasting ML models"""
    
    MODEL_TYPES = [
        ('random_forest', 'Random Forest'),
        ('linear_regression', 'Linear Regression'),
        ('neural_network', 'Neural Network'),
        ('ensemble', 'Ensemble Model'),
    ]
    
    STATUS_CHOICES = [
        ('training', 'Training'),
        ('ready', 'Ready'),
        ('deprecated', 'Deprecated'),
        ('error', 'Error'),
    ]
    
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES, default='random_forest')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training')
    
    # Model configuration
    pest_type = models.ForeignKey(PestType, on_delete=models.CASCADE)
    model_file = models.FileField(upload_to='forecasting_models/', blank=True)
    model_path = models.CharField(max_length=500, blank=True)
    
    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)  # Root Mean Square Error
    mae = models.FloatField(null=True, blank=True)   # Mean Absolute Error
    r2_score = models.FloatField(null=True, blank=True)  # R-squared
    
    # Training information
    training_data_points = models.IntegerField(default=0)
    features_used = models.JSONField(default=list)  # List of weather features used
    training_period_start = models.DateField(null=True, blank=True)
    training_period_end = models.DateField(null=True, blank=True)
    
    # Model metadata
    version = models.CharField(max_length=50, default='1.0')
    is_active = models.BooleanField(default=False)
    
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.pest_type.name}"
    
    def save(self, *args, **kwargs):
        # Ensure only one model per pest type is active
        if self.is_active:
            PestForecastModel.objects.filter(
                pest_type=self.pest_type, 
                is_active=True
            ).exclude(id=self.id).update(is_active=False)
        super().save(*args, **kwargs)


class PestForecast(models.Model):
    """Model for storing pest outbreak forecasts"""
    
    RISK_LEVELS = [
        ('low', 'Low Risk'),
        ('moderate', 'Moderate Risk'),
        ('high', 'High Risk'),
        ('critical', 'Critical Risk'),
    ]
    
    FORECAST_TYPES = [
        ('daily', 'Daily Forecast'),
        ('weekly', 'Weekly Forecast'),
        ('monthly', 'Monthly Forecast'),
        ('seasonal', 'Seasonal Forecast'),
    ]
    
    # Forecast identification
    forecast_id = models.UUIDField(default=uuid.uuid4, unique=True)
    pest_type = models.ForeignKey(PestType, on_delete=models.CASCADE)
    model_used = models.ForeignKey(PestForecastModel, on_delete=models.SET_NULL, null=True)
    
    # Location and timing
    location_name = models.CharField(max_length=200)
    latitude = models.DecimalField(max_digits=10, decimal_places=8)
    longitude = models.DecimalField(max_digits=11, decimal_places=8)
    
    forecast_date = models.DateField()  # Date this forecast was made
    target_date = models.DateField()    # Date the forecast is for
    forecast_type = models.CharField(max_length=20, choices=FORECAST_TYPES, default='daily')
    
    # Forecast results
    risk_level = models.CharField(max_length=20, choices=RISK_LEVELS)
    risk_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    confidence = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    
    # Predicted pest activity
    predicted_pest_count = models.IntegerField(default=0)
    predicted_outbreak_probability = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)]
    )
    
    # Weather conditions used for forecast
    weather_conditions = models.JSONField(default=dict)
    
    # Recommendations
    management_recommendations = models.TextField(blank=True)
    monitoring_recommendations = models.TextField(blank=True)
    treatment_recommendations = models.TextField(blank=True)
    
    # Validation and accuracy
    actual_outcome = models.CharField(max_length=20, choices=RISK_LEVELS, blank=True, null=True)
    actual_pest_count = models.IntegerField(null=True, blank=True)
    forecast_accuracy = models.FloatField(null=True, blank=True)
    
    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['pest_type', 'location_name', 'target_date', 'forecast_type']
        ordering = ['-target_date', '-created_at']
    
    def __str__(self):
        return f"{self.pest_type.name} - {self.location_name} - {self.target_date} ({self.risk_level})"


class ForecastAlert(models.Model):
    """Model for managing pest outbreak alerts"""
    
    ALERT_TYPES = [
        ('outbreak_warning', 'Outbreak Warning'),
        ('high_risk', 'High Risk Alert'),
        ('monitoring', 'Monitoring Alert'),
        ('treatment_reminder', 'Treatment Reminder'),
    ]
    
    ALERT_STATUS = [
        ('active', 'Active'),
        ('acknowledged', 'Acknowledged'),
        ('resolved', 'Resolved'),
        ('expired', 'Expired'),
    ]
    
    # Alert identification
    alert_id = models.UUIDField(default=uuid.uuid4, unique=True)
    alert_type = models.CharField(max_length=30, choices=ALERT_TYPES)
    status = models.CharField(max_length=20, choices=ALERT_STATUS, default='active')
    
    # Related forecast
    forecast = models.ForeignKey(PestForecast, on_delete=models.CASCADE, related_name='alerts')
    
    # Alert details
    title = models.CharField(max_length=200)
    message = models.TextField()
    priority = models.IntegerField(default=5, validators=[MinValueValidator(1), MaxValueValidator(10)])
    
    # Timing
    valid_from = models.DateTimeField()
    valid_until = models.DateTimeField()
    
    # Targeting
    target_users = models.ManyToManyField(User, blank=True)
    target_locations = models.JSONField(default=list)  # List of location names
    
    # Actions taken
    acknowledged_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, 
        related_name='acknowledged_alerts'
    )
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-priority', '-created_at']
    
    def __str__(self):
        return f"{self.alert_type} - {self.title}"


class ForecastFeedback(models.Model):
    """Model for collecting feedback on forecast accuracy"""
    
    FEEDBACK_TYPES = [
        ('accurate', 'Accurate Forecast'),
        ('overestimated', 'Overestimated Risk'),
        ('underestimated', 'Underestimated Risk'),
        ('timing_off', 'Timing was Off'),
        ('conditions_changed', 'Weather Conditions Changed'),
    ]
    
    forecast = models.ForeignKey(PestForecast, on_delete=models.CASCADE, related_name='feedback')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
    feedback_type = models.CharField(max_length=30, choices=FEEDBACK_TYPES)
    
    # Observed conditions
    observed_pest_count = models.IntegerField()
    observed_weather_conditions = models.JSONField(default=dict, blank=True)
    
    # Feedback details
    accuracy_rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="1=Very Inaccurate, 5=Very Accurate"
    )
    comments = models.TextField(blank=True)
    
    # Location validation
    latitude = models.DecimalField(max_digits=10, decimal_places=8, null=True, blank=True)
    longitude = models.DecimalField(max_digits=11, decimal_places=8, null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback for {self.forecast.forecast_id}: {self.feedback_type}"


class ForecastingAnalytics(models.Model):
    """Model for storing aggregated forecasting analytics"""
    
    date = models.DateField()
    pest_type = models.ForeignKey(PestType, on_delete=models.CASCADE)
    location_name = models.CharField(max_length=200)
    
    # Daily forecast statistics
    total_forecasts = models.IntegerField(default=0)
    high_risk_forecasts = models.IntegerField(default=0)
    critical_risk_forecasts = models.IntegerField(default=0)
    
    # Accuracy metrics
    avg_forecast_accuracy = models.FloatField(null=True, blank=True)
    total_feedback_count = models.IntegerField(default=0)
    
    # Alert statistics
    alerts_generated = models.IntegerField(default=0)
    alerts_acknowledged = models.IntegerField(default=0)
    
    # Weather summary
    avg_temperature = models.FloatField(null=True, blank=True)
    avg_humidity = models.FloatField(null=True, blank=True)
    total_rainfall = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['date', 'pest_type', 'location_name']
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.pest_type.name} - {self.location_name} - {self.date}"


class HistoricalPestData(models.Model):
    """Model for storing historical pest occurrence data for training forecasting models"""
    
    pest_type = models.ForeignKey(PestType, on_delete=models.CASCADE)
    
    # Location and timing
    location_name = models.CharField(max_length=200)
    latitude = models.DecimalField(max_digits=10, decimal_places=8)
    longitude = models.DecimalField(max_digits=11, decimal_places=8)
    recorded_date = models.DateField()
    
    # Pest occurrence data
    pest_count = models.IntegerField(default=0)
    infestation_level = models.CharField(
        max_length=20,
        choices=[
            ('none', 'No Infestation'),
            ('low', 'Low Infestation'),
            ('moderate', 'Moderate Infestation'),
            ('high', 'High Infestation'),
            ('severe', 'Severe Infestation'),
        ],
        default='none'
    )
    
    # Associated weather conditions
    temperature = models.FloatField()
    humidity = models.FloatField()
    rainfall_7days = models.FloatField(default=0.0)
    rainfall_14days = models.FloatField(default=0.0)
    
    # Data source
    data_source = models.CharField(max_length=100, default='Manual Entry')
    verified = models.BooleanField(default=False)
    
    # Metadata
    notes = models.TextField(blank=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['pest_type', 'location_name', 'recorded_date']
        ordering = ['-recorded_date']
    
    def __str__(self):
        return f"{self.pest_type.name} - {self.location_name} - {self.recorded_date}"