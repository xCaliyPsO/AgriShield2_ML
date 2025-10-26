from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for ViewSets
router = DefaultRouter()
router.register('weather', views.WeatherDataViewSet)
router.register('models', views.PestForecastModelViewSet)
router.register('forecasts', views.PestForecastViewSet)
router.register('alerts', views.ForecastAlertViewSet)
router.register('feedback', views.ForecastFeedbackViewSet)
router.register('historical-data', views.HistoricalPestDataViewSet)

app_name = 'pest_forecasting'

urlpatterns = [
    # Main forecasting API endpoints
    path('forecast/', views.PestForecastingAPIView.as_view(), name='generate_forecast'),
    path('quick-forecast/', views.quick_forecast, name='quick_forecast'),
    
    # ViewSet routes
    path('api/', include(router.urls)),
]
