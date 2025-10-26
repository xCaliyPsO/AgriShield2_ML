# AgriShield ML - Django Framework

## Overview

Agricultural pest detection and forecasting system built with Django and YOLO v8.
Detects 4 rice pest types: Rice Bug, Black Bug, Brown Hopper, Green Hopper.

## Features

- **Pest Detection**: Real-time pest detection using YOLO v8
- **Model Training**: Admin can train new YOLO models via web interface
- **REST API**: Django REST Framework with authentication
- **Admin Interface**: Complete web-based management interface
- **Forecasting**: Weather-based pest outbreak predictions
- **Database**: MySQL with proper relationships

## Quick Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Database**
   ```bash
   python setup.py
   ```

3. **Run Server**
   ```bash
   python manage.py runserver
   ```

## API Endpoints

- `GET /health/` - Health check
- `POST /detect/` - Detect pests in image
- `GET /admin/` - Admin interface
- `GET /api/` - REST API endpoints

## Usage

### Pest Detection
```bash
curl -X POST http://localhost:8000/detect/ -F "image=@photo.jpg"
```

### Admin Access
Navigate to `http://localhost:8000/admin/` and login with superuser credentials.

## Model Training

### Train New Models (Admin Only)
1. **Access Admin**: `http://localhost:8000/admin/`
2. **Create Dataset**: ML Training → Datasets → Add new
3. **Create Training Job**: ML Training → Training jobs → Add new
4. **Configure Parameters**: Set epochs, batch size, model architecture
5. **Start Training**: Select job → Actions → Start training
6. **Monitor Progress**: View real-time training metrics

### Training API
```bash
# Create training job
POST /api/ml-training/jobs/create_job/
{
  "name": "New Rice Pest Model",
  "training_type": "object_detection",
  "model_architecture": "yolov8n",
  "dataset_id": 1,
  "epochs": 50
}

# Start training
POST /api/ml-training/jobs/{id}/start/

# Monitor progress
GET /api/ml-training/jobs/{id}/metrics/
```

## Technologies

- Django 5.1+
- Django REST Framework
- YOLO v8 (Ultralytics)
- MySQL
- Python 3.8+

## Model Performance

- **Accuracy**: 99.5% mAP@0.5
- **Inference Time**: ~50ms per image
- **Model Size**: 6MB

## Project Structure

```
├── agrihield_django/    # Django project settings
├── pest_detection/      # Pest detection app
├── pest_forecasting/    # Forecasting app
├── ml_training/         # Training management app
├── ml_models/          # Trained YOLO models
├── manage.py           # Django management
└── requirements.txt    # Dependencies
```

## Requirements

- Python 3.8+
- MySQL/MariaDB
- CUDA GPU (optional, for training)

## License

MIT License