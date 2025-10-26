# Local Server Deployment Commands

## Prerequisites
- Python 3.8+
- MySQL/MariaDB database
- Git (optional)

## Step-by-Step Deployment

### 1. Download/Clone Repository
```bash
# Option A: Download ZIP and extract
# Option B: Git clone
git clone https://github.com/YOUR_USERNAME/AgriShield-ML-Django.git
cd AgriShield-ML-Django
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Database Setup (MySQL)
```bash
# Create database in MySQL
mysql -u root -p
CREATE DATABASE asdb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;
```

### 4. Django Setup (Automated)
```bash
python setup.py
```
This will:
- Create database tables
- Prompt for admin username/password
- Setup initial data

### 5. Start Local Server
```bash
python manage.py runserver
```

### 6. Access Application
- **Web Interface**: http://localhost:8000/admin/
- **API Health Check**: http://localhost:8000/health/
- **Pest Detection API**: http://localhost:8000/detect/

## Manual Commands (Alternative)

If `setup.py` fails, run manually:
```bash
# Create database migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Start server
python manage.py runserver
```

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health/
```

### Pest Detection
```bash
curl -X POST http://localhost:8000/detect/ -F "image=@test_image.jpg"
```

## Production Deployment

For production server:
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn agrihield_django.wsgi:application --bind 0.0.0.0:8000
```

## Troubleshooting

### Database Connection Error
1. Ensure MySQL is running
2. Check database credentials in `agrihield_django/settings.py`
3. Create database: `CREATE DATABASE asdb;`

### Module Import Error  
1. Ensure virtual environment is activated
2. Install requirements: `pip install -r requirements.txt`

### Model Loading Error
1. Check model file exists: `ml_models/pest_detection/best.pt`
2. Verify file permissions

## File Structure
```
AgriShield_ML_Minimal/
├── manage.py              # Django management
├── requirements.txt       # Dependencies
├── setup.py              # Automated setup
├── agrihield_django/     # Django project
├── pest_detection/       # Main app
├── pest_forecasting/     # Forecasting
├── ml_training/          # Training
└── ml_models/           # YOLO model
```

## Support
All files needed for deployment are included in this folder.
No additional setup or configuration required.
