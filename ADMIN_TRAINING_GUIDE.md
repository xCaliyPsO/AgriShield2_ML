# 🎯 Admin Training Guide

## How Admins Can Train New YOLO Models

Your Django admin interface has **complete training capabilities** for creating new pest detection models.

## 🔧 **Training Setup (Admin Access)**

### **1. Access Admin Interface**
```
http://localhost:8000/admin/
Login with superuser credentials
```

### **2. Navigate to ML Training Section**
- **Datasets** - Manage training datasets
- **Training jobs** - Create and monitor training jobs
- **Training metrics** - View training progress
- **Model checkpoints** - Manage saved models

## 📊 **Step-by-Step Training Process**

### **Step 1: Prepare Dataset**
1. Go to **"Datasets"** in admin
2. Click **"Add Dataset"**
3. Configure:
   ```
   Name: "Rice Pest Dataset v2"
   Type: "Object Detection" 
   Status: "Ready"
   Classes: ["Rice_Bug", "black-bug", "brown_hopper", "green_hopper"]
   Train/Val/Test splits: 70%/20%/10%
   ```

### **Step 2: Create Training Job**
1. Go to **"Training jobs"** in admin
2. Click **"Add Training Job"**
3. Configure training parameters:
   ```
   Name: "YOLO v8 Rice Pests - Improved"
   Training type: "Object Detection"
   Model architecture: "yolov8n" (or yolov8s, yolov8m)
   Dataset: Select your dataset
   
   Training Parameters:
   - Epochs: 50-100
   - Batch size: 8-16
   - Learning rate: 0.001
   - Image size: 640
   ```

### **Step 3: Start Training**
1. **Via Admin Interface:**
   - Go to training job
   - Click **"Start"** action
   - Monitor progress in real-time

2. **Via API:**
   ```bash
   POST /api/ml-training/jobs/{job_id}/start/
   ```

3. **Via Command Line:**
   ```bash
   python manage.py start_training {job_id}
   ```

### **Step 4: Monitor Training**
1. **View Progress:**
   - Training jobs → Select job → View current epoch/progress
   - Training metrics → See loss/accuracy curves
   - Training logs → Real-time training output

2. **Check Status:**
   ```bash
   GET /api/ml-training/jobs/{job_id}/
   GET /api/ml-training/jobs/{job_id}/metrics/
   GET /api/ml-training/jobs/{job_id}/logs/
   ```

### **Step 5: Deploy Trained Model**
1. **After Training Completes:**
   - New model appears in **"Detection models"**
   - Automatically saved to `ml_models/pest_detection/`
   - Ready for pest detection API

2. **Activate New Model:**
   - Go to **"Detection models"** 
   - Find your new model
   - Click **"Activate"** to make it the active detection model

## 🤖 **Training Features Available**

### **Model Architectures:**
```
- yolov8n (Nano)   - Fast, smaller
- yolov8s (Small)  - Balanced
- yolov8m (Medium) - More accurate
- yolov8l (Large)  - High accuracy
- yolov8x (XLarge) - Best accuracy
```

### **Training Types:**
```
- Object Detection  - Your main use case
- Classification    - Simple classification
- Fine Tuning      - Improve existing model
```

### **Advanced Settings:**
```
- Data Augmentation: Enabled by default
- Pretrained: Uses COCO pretrained weights
- Mixed Precision: Available for GPU training
- Early Stopping: Automatic if validation stops improving
```

## 📈 **Real-Time Monitoring**

### **Training Progress:**
- **Current epoch** / Total epochs
- **Progress percentage**
- **Training loss** decreasing
- **Validation accuracy** increasing
- **Estimated completion time**

### **Performance Metrics:**
- **mAP@0.5** - Main accuracy metric
- **Precision/Recall** per class
- **Training/Validation loss**
- **Inference speed** (FPS)

### **Resource Monitoring:**
- **CPU/GPU usage**
- **Memory consumption**
- **Training time per epoch**
- **Model file size**

## 🎯 **Training Best Practices**

### **For Better Models:**
1. **More Data**: Add more training images
2. **Balanced Classes**: Equal samples per pest type
3. **Quality Annotations**: Accurate bounding boxes
4. **Data Augmentation**: Enabled by default
5. **Proper Validation**: Use separate validation set

### **Training Parameters:**
```python
# For fast experimentation
epochs = 25, batch_size = 16, model = "yolov8n"

# For production models  
epochs = 100, batch_size = 8, model = "yolov8m"

# For research/best accuracy
epochs = 200, batch_size = 4, model = "yolov8x"
```

## 🚀 **Training Commands Summary**

### **Admin GUI Training:**
```
1. http://localhost:8000/admin/
2. ML Training → Training jobs → Add
3. Configure parameters → Save
4. Select job → Actions → Start training
5. Monitor progress in real-time
```

### **API Training:**
```bash
# Create job
POST /api/ml-training/jobs/create_job/
{
  "name": "New Model",
  "dataset_id": 1,
  "epochs": 50,
  "model_architecture": "yolov8n"
}

# Start training
POST /api/ml-training/jobs/{id}/start/

# Monitor progress
GET /api/ml-training/jobs/{id}/metrics/
```

### **Command Line Training:**
```bash
# Start specific job
python manage.py start_training 1

# View training stats  
python manage.py shell
>>> from ml_training.models import TrainingJob
>>> job = TrainingJob.objects.get(id=1)
>>> print(f"Progress: {job.progress_percentage}%")
```

## ✅ **Training System Status**

**Your minimal folder includes:**
- ✅ **Complete admin interface** for training management
- ✅ **YOLO training engine** connected to Django
- ✅ **Real-time progress monitoring**
- ✅ **Automatic model deployment** after training
- ✅ **Training metrics and logging**
- ✅ **Model version management**

**Admin can train new models with just a few clicks!** 🎯
