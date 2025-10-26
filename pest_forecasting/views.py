import json
import logging
import requests
from datetime import datetime, timedelta, date
from typing import Dict, List, Any

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.db.models import Q, Count, Avg, Sum
from django.shortcuts import get_object_or_404

from rest_framework import status, viewsets
from rest_framework.decorators import api_view, action, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from pest_detection.models import PestType
from .models import (
    WeatherData, PestForecastModel, PestForecast, ForecastAlert,
    ForecastFeedback, ForecastingAnalytics, HistoricalPestData
)
from .serializers import (
    WeatherDataSerializer, PestForecastModelSerializer, PestForecastSerializer,
    ForecastAlertSerializer, ForecastFeedbackSerializer, ForecastingAnalyticsSerializer,
    HistoricalPestDataSerializer, ForecastRequestSerializer, QuickForecastSerializer,
    ForecastResultSerializer, WeatherForecastSerializer
)

# Import forecasting engine
import sys
from pathlib import Path
sys.path.append(str(Path(settings.BASE_DIR) / 'pest_forecasting_system' / 'engines'))

try:
    from pest_forecasting_engine import PestForecastingEngine
    FORECASTING_ENGINE_AVAILABLE = True
except ImportError:
    FORECASTING_ENGINE_AVAILABLE = False

logger = logging.getLogger('pest_forecasting')


class WeatherDataViewSet(viewsets.ModelViewSet):
    """ViewSet for managing weather data"""
    
    queryset = WeatherData.objects.all()
    serializer_class = WeatherDataSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by location if provided
        location = self.request.query_params.get('location', None)
        if location:
            queryset = queryset.filter(location_name__icontains=location)
        
        # Filter by date range
        date_from = self.request.query_params.get('date_from', None)
        date_to = self.request.query_params.get('date_to', None)
        
        if date_from:
            queryset = queryset.filter(recorded_at__gte=date_from)
        if date_to:
            queryset = queryset.filter(recorded_at__lte=date_to)
        
        return queryset.order_by('-recorded_at')
    
    @action(detail=False, methods=['get'])
    def current(self, request):
        """Get current weather for a location"""
        location = request.query_params.get('location')
        if not location:
            return Response(
                {'error': 'Location parameter is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get most recent weather data for location
        weather = self.get_queryset().filter(
            location_name=location,
            is_forecast=False
        ).first()
        
        if not weather:
            return Response(
                {'error': f'No weather data found for {location}'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer = self.get_serializer(weather)
        return Response(serializer.data)


class PestForecastModelViewSet(viewsets.ModelViewSet):
    """ViewSet for managing pest forecast models"""
    
    queryset = PestForecastModel.objects.all()
    serializer_class = PestForecastModelSerializer
    permission_classes = [IsAuthenticated]
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Activate a specific forecasting model"""
        model = self.get_object()
        
        # Deactivate other models for this pest type
        PestForecastModel.objects.filter(
            pest_type=model.pest_type, is_active=True
        ).update(is_active=False)
        
        # Activate this model
        model.is_active = True
        model.save()
        
        return Response({'status': 'Model activated successfully'})
    
    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get all active forecasting models"""
        models = self.get_queryset().filter(is_active=True)
        serializer = self.get_serializer(models, many=True)
        return Response(serializer.data)


class PestForecastViewSet(viewsets.ModelViewSet):
    """ViewSet for managing pest forecasts"""
    
    queryset = PestForecast.objects.all()
    serializer_class = PestForecastSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by location
        location = self.request.query_params.get('location', None)
        if location:
            queryset = queryset.filter(location_name__icontains=location)
        
        # Filter by pest type
        pest_type = self.request.query_params.get('pest_type', None)
        if pest_type:
            queryset = queryset.filter(pest_type_id=pest_type)
        
        # Filter by risk level
        risk_level = self.request.query_params.get('risk_level', None)
        if risk_level:
            queryset = queryset.filter(risk_level=risk_level)
        
        # Filter by date range
        date_from = self.request.query_params.get('date_from', None)
        date_to = self.request.query_params.get('date_to', None)
        
        if date_from:
            queryset = queryset.filter(target_date__gte=date_from)
        if date_to:
            queryset = queryset.filter(target_date__lte=date_to)
        
        return queryset.order_by('-target_date', '-created_at')
    
    @action(detail=False, methods=['get'])
    def current(self, request):
        """Get current active forecasts"""
        today = timezone.now().date()
        forecasts = self.get_queryset().filter(
            target_date__gte=today,
            target_date__lte=today + timedelta(days=7)
        )
        
        serializer = self.get_serializer(forecasts, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def high_risk(self, request):
        """Get high and critical risk forecasts"""
        forecasts = self.get_queryset().filter(
            risk_level__in=['high', 'critical'],
            target_date__gte=timezone.now().date()
        )
        
        serializer = self.get_serializer(forecasts, many=True)
        return Response(serializer.data)


class PestForecastingAPIView(APIView):
    """Main forecasting API for generating new forecasts"""
    
    permission_classes = [AllowAny]
    
    def post(self, request):
        """Generate pest forecast for specific conditions"""
        serializer = ForecastRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        try:
            # Get pest type
            pest_type = get_object_or_404(PestType, id=data['pest_type_id'])
            
            # Get active model for this pest type
            model = PestForecastModel.objects.filter(
                pest_type=pest_type, is_active=True
            ).first()
            
            if not model:
                return Response({
                    'error': f'No active forecasting model found for {pest_type.name}'
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Generate forecast
            forecast_result = self.generate_forecast(pest_type, model, data)
            
            return Response(forecast_result, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def generate_forecast(self, pest_type, model, data):
        """Generate forecast using ML model or heuristics"""
        
        # Create weather data record
        weather_data = WeatherData.objects.create(
            location_name=data['location_name'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            recorded_at=timezone.now(),
            temperature=data['temperature'],
            humidity=data['humidity'],
            pressure=data['pressure'],
            wind_speed=data['wind_speed'],
            rainfall_1h=data['rainfall_1h'],
            rainfall_24h=data['rainfall_24h'],
            cloudiness=data['cloudiness'],
            data_source='API_Request',
            is_forecast=False
        )
        
        # Use forecasting engine if available
        if FORECASTING_ENGINE_AVAILABLE:
            try:
                engine = PestForecastingEngine()
                forecast_data = engine.generate_forecast(
                    pest_type.name.lower().replace(' ', '_'),
                    data['location_name'],
                    data['latitude'],
                    data['longitude'],
                    weather_conditions={
                        'temperature': data['temperature'],
                        'humidity': data['humidity'],
                        'rainfall_1h': data['rainfall_1h'],
                        'wind_speed': data['wind_speed'],
                        'pressure': data['pressure']
                    }
                )
                
                risk_level = forecast_data.get('risk_level', 'low')
                risk_score = forecast_data.get('risk_score', 0.0)
                confidence = forecast_data.get('confidence', 0.5)
                predicted_count = forecast_data.get('predicted_count', 0)
                
            except Exception as e:
                logger.warning(f"Forecasting engine failed, using heuristics: {e}")
                # Fallback to simple heuristic-based forecasting
                risk_level, risk_score, confidence, predicted_count = self.heuristic_forecast(
                    pest_type, data
                )
        else:
            # Use simple heuristic-based forecasting
            risk_level, risk_score, confidence, predicted_count = self.heuristic_forecast(
                pest_type, data
            )
        
        # Save forecast to database
        forecast = PestForecast.objects.create(
            pest_type=pest_type,
            model_used=model,
            location_name=data['location_name'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            forecast_date=timezone.now().date(),
            target_date=timezone.now().date() + timedelta(days=1),
            forecast_type='daily',
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=confidence,
            predicted_pest_count=predicted_count,
            predicted_outbreak_probability=risk_score,
            weather_conditions={
                'temperature': data['temperature'],
                'humidity': data['humidity'],
                'rainfall_1h': data['rainfall_1h'],
                'rainfall_24h': data['rainfall_24h'],
                'wind_speed': data['wind_speed'],
                'pressure': data['pressure'],
                'cloudiness': data['cloudiness']
            },
            management_recommendations=self.get_management_recommendations(pest_type, risk_level),
            monitoring_recommendations=self.get_monitoring_recommendations(pest_type, risk_level),
            treatment_recommendations=pest_type.pesticide_recommendations or "",
            created_by=self.request.user if self.request.user.is_authenticated else None
        )
        
        # Generate alerts if high risk
        if risk_level in ['high', 'critical']:
            self.create_alert(forecast)
        
        # Prepare response
        return {
            'status': 'success',
            'forecast_id': forecast.forecast_id,
            'location_name': forecast.location_name,
            'pest_forecasts': [{
                'pest_type': pest_type.name,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'confidence': confidence,
                'predicted_count': predicted_count
            }],
            'risk_summary': {
                'overall_risk': risk_level,
                'max_risk_score': risk_score,
                'total_predicted_pests': predicted_count
            },
            'weather_conditions': forecast.weather_conditions,
            'management_recommendations': [forecast.management_recommendations] if forecast.management_recommendations else [],
            'monitoring_recommendations': [forecast.monitoring_recommendations] if forecast.monitoring_recommendations else [],
            'treatment_recommendations': [forecast.treatment_recommendations] if forecast.treatment_recommendations else [],
            'model_used': model.name,
            'forecast_date': forecast.forecast_date,
            'confidence': confidence
        }
    
    def heuristic_forecast(self, pest_type, data):
        """Simple heuristic-based forecasting"""
        
        # Basic risk assessment based on weather conditions
        risk_factors = 0
        
        # Temperature risk
        if 20 <= data['temperature'] <= 35:
            risk_factors += 2
        elif 15 <= data['temperature'] <= 40:
            risk_factors += 1
        
        # Humidity risk
        if data['humidity'] >= 70:
            risk_factors += 2
        elif data['humidity'] >= 60:
            risk_factors += 1
        
        # Rainfall risk
        if data['rainfall_24h'] > 0:
            risk_factors += 1
        
        # Wind speed (lower wind = higher risk for some pests)
        if data['wind_speed'] < 5:
            risk_factors += 1
        
        # Calculate risk level
        if risk_factors >= 5:
            risk_level = 'critical'
            risk_score = 0.8 + (risk_factors - 5) * 0.05
        elif risk_factors >= 3:
            risk_level = 'high'
            risk_score = 0.6 + (risk_factors - 3) * 0.1
        elif risk_factors >= 2:
            risk_level = 'moderate'
            risk_score = 0.3 + (risk_factors - 2) * 0.15
        else:
            risk_level = 'low'
            risk_score = risk_factors * 0.15
        
        # Ensure risk_score is within bounds
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Estimate pest count based on risk
        predicted_count = int(risk_score * 20)  # Max 20 pests
        
        # Confidence based on data completeness
        confidence = 0.6  # Moderate confidence for heuristic method
        
        return risk_level, risk_score, confidence, predicted_count
    
    def get_management_recommendations(self, pest_type, risk_level):
        """Get management recommendations based on risk level"""
        if risk_level == 'critical':
            return f"IMMEDIATE ACTION REQUIRED: High {pest_type.name} activity predicted. Implement emergency control measures."
        elif risk_level == 'high':
            return f"HIGH ALERT: Increased {pest_type.name} activity expected. Prepare control measures and monitor closely."
        elif risk_level == 'moderate':
            return f"MODERATE RISK: Some {pest_type.name} activity possible. Continue regular monitoring."
        else:
            return f"LOW RISK: Minimal {pest_type.name} activity expected. Maintain routine monitoring."
    
    def get_monitoring_recommendations(self, pest_type, risk_level):
        """Get monitoring recommendations based on risk level"""
        if risk_level in ['critical', 'high']:
            return f"Increase monitoring frequency to twice daily. Focus on early detection of {pest_type.name}."
        elif risk_level == 'moderate':
            return f"Monitor daily for {pest_type.name} activity. Check vulnerable crop stages."
        else:
            return f"Continue routine weekly monitoring for {pest_type.name}."
    
    def create_alert(self, forecast):
        """Create alert for high-risk forecasts"""
        alert_type = 'outbreak_warning' if forecast.risk_level == 'critical' else 'high_risk'
        
        ForecastAlert.objects.create(
            alert_type=alert_type,
            forecast=forecast,
            title=f"{forecast.pest_type.name} {forecast.risk_level.upper()} Risk Alert",
            message=f"High {forecast.pest_type.name} activity predicted for {forecast.location_name} on {forecast.target_date}",
            priority=9 if forecast.risk_level == 'critical' else 7,
            valid_from=timezone.now(),
            valid_until=timezone.now() + timedelta(days=3),
            target_locations=[forecast.location_name]
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def quick_forecast(request):
    """Quick forecast endpoint for multiple pest types"""
    
    serializer = QuickForecastSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    
    try:
        # Get pest types to forecast
        if 'pest_type_ids' in data:
            pest_types = PestType.objects.filter(id__in=data['pest_type_ids'])
        else:
            pest_types = PestType.objects.all()
        
        # Get or fetch weather data
        weather_data = None
        if data.get('use_current_weather', True):
            # Try to get recent weather data for location
            weather_data = WeatherData.objects.filter(
                location_name=data['location_name'],
                recorded_at__gte=timezone.now() - timedelta(hours=6)
            ).first()
        
        if not weather_data and not data.get('use_current_weather', True):
            # Use provided weather data
            weather_conditions = {
                'temperature': data.get('temperature', 25),
                'humidity': data.get('humidity', 70),
                'rainfall_24h': data.get('rainfall_24h', 0),
                'wind_speed': 5.0,
                'pressure': 1013.0,
                'cloudiness': 50.0
            }
        elif weather_data:
            weather_conditions = {
                'temperature': weather_data.temperature,
                'humidity': weather_data.humidity,
                'rainfall_24h': weather_data.rainfall_24h,
                'wind_speed': weather_data.wind_speed,
                'pressure': weather_data.pressure,
                'cloudiness': weather_data.cloudiness
            }
        else:
            # Default weather conditions
            weather_conditions = {
                'temperature': 25,
                'humidity': 70,
                'rainfall_24h': 0,
                'wind_speed': 5.0,
                'pressure': 1013.0,
                'cloudiness': 50.0
            }
        
        # Generate forecasts for all pest types
        forecasts = []
        max_risk_score = 0
        overall_risk = 'low'
        
        for pest_type in pest_types:
            # Simple heuristic forecast
            risk_factors = 0
            temp = weather_conditions['temperature']
            humidity = weather_conditions['humidity']
            rainfall = weather_conditions['rainfall_24h']
            
            # Temperature assessment
            if 20 <= temp <= 35:
                risk_factors += 2
            elif 15 <= temp <= 40:
                risk_factors += 1
            
            # Humidity assessment
            if humidity >= 70:
                risk_factors += 2
            elif humidity >= 60:
                risk_factors += 1
            
            # Rainfall assessment
            if rainfall > 0:
                risk_factors += 1
            
            # Calculate risk
            if risk_factors >= 5:
                risk_level = 'critical'
                risk_score = 0.8
            elif risk_factors >= 3:
                risk_level = 'high'
                risk_score = 0.6
            elif risk_factors >= 2:
                risk_level = 'moderate'
                risk_score = 0.4
            else:
                risk_level = 'low'
                risk_score = 0.2
            
            forecasts.append({
                'pest_type': pest_type.name,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'confidence': 0.6,
                'predicted_count': int(risk_score * 15)
            })
            
            if risk_score > max_risk_score:
                max_risk_score = risk_score
                overall_risk = risk_level
        
        return Response({
            'status': 'success',
            'location_name': data['location_name'],
            'pest_forecasts': forecasts,
            'risk_summary': {
                'overall_risk': overall_risk,
                'max_risk_score': max_risk_score
            },
            'weather_conditions': weather_conditions,
            'forecast_date': timezone.now().date()
        })
        
    except Exception as e:
        logger.error(f"Quick forecast failed: {e}")
        return Response({
            'status': 'error',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Additional ViewSets
class ForecastAlertViewSet(viewsets.ModelViewSet):
    """ViewSet for managing forecast alerts"""
    
    queryset = ForecastAlert.objects.all()
    serializer_class = ForecastAlertSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by status
        status_filter = self.request.query_params.get('status', None)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Filter by alert type
        alert_type = self.request.query_params.get('alert_type', None)
        if alert_type:
            queryset = queryset.filter(alert_type=alert_type)
        
        return queryset.order_by('-priority', '-created_at')
    
    @action(detail=True, methods=['post'])
    def acknowledge(self, request, pk=None):
        """Acknowledge an alert"""
        alert = self.get_object()
        alert.status = 'acknowledged'
        alert.acknowledged_by = request.user
        alert.acknowledged_at = timezone.now()
        alert.save()
        
        return Response({'status': 'Alert acknowledged'})
    
    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get active alerts"""
        alerts = self.get_queryset().filter(
            status='active',
            valid_until__gte=timezone.now()
        )
        
        serializer = self.get_serializer(alerts, many=True)
        return Response(serializer.data)


class ForecastFeedbackViewSet(viewsets.ModelViewSet):
    """ViewSet for managing forecast feedback"""
    
    queryset = ForecastFeedback.objects.all()
    serializer_class = ForecastFeedbackSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class HistoricalPestDataViewSet(viewsets.ModelViewSet):
    """ViewSet for managing historical pest data"""
    
    queryset = HistoricalPestData.objects.all()
    serializer_class = HistoricalPestDataSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)