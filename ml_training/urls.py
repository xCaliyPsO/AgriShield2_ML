from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for ViewSets
router = DefaultRouter()
router.register('datasets', views.DatasetViewSet)
router.register('jobs', views.TrainingJobViewSet)
router.register('metrics', views.TrainingMetricViewSet)
router.register('logs', views.TrainingLogViewSet)
router.register('checkpoints', views.ModelCheckpointViewSet)

app_name = 'ml_training'

urlpatterns = [
    # Training statistics
    path('stats/', views.training_stats, name='training_stats'),
    
    # ViewSet routes
    path('api/', include(router.urls)),
]
