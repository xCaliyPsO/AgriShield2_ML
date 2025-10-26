from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APITestCase
from rest_framework import status
from .models import PestType, DetectionModel


class PestDetectionTestCase(TestCase):
    """Test cases for pest detection models"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        self.pest_type = PestType.objects.create(
            name='Rice_Bug',
            scientific_name='Leptocorisa oratorius',
            description='Test pest type'
        )
    
    def test_pest_type_creation(self):
        """Test pest type model creation"""
        self.assertEqual(self.pest_type.name, 'Rice_Bug')
        self.assertEqual(str(self.pest_type), 'Rice_Bug')
    
    def test_detection_model_creation(self):
        """Test detection model creation"""
        model = DetectionModel.objects.create(
            name='Test Model',
            model_type='yolo',
            version='1.0',
            status='active',
            created_by=self.user
        )
        
        self.assertEqual(model.name, 'Test Model')
        self.assertEqual(str(model), 'Test Model v1.0')


class PestDetectionAPITestCase(APITestCase):
    """Test cases for pest detection API"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com', 
            password='testpass123'
        )
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)
    
    def test_stats_endpoint(self):
        """Test stats endpoint"""
        response = self.client.get('/stats/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('total_detections', response.data)
