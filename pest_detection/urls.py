from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for ViewSets
router = DefaultRouter()
router.register('pest-types', views.PestTypeViewSet)
router.register('models', views.DetectionModelViewSet)
router.register('sessions', views.DetectionSessionViewSet)
router.register('detections', views.PestDetectionViewSet)
router.register('feedback', views.DetectionFeedbackViewSet)

app_name = 'pest_detection'

urlpatterns = [
    # API endpoints (equivalent to Flask routes)
    path('health/', views.health_check, name='health_check'),
    path('detect/', views.PestDetectionAPIView.as_view(), name='detect'),
    path('stats/', views.detection_stats, name='stats'),
    
    # ViewSet routes
    path('api/', include(router.urls)),
]
