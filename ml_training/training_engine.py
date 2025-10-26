#!/usr/bin/env python3
"""
Django Training Engine - Connects to Original Training Scripts
Integrates your existing YOLO training with Django admin interface
"""

import os
import sys
import subprocess
import logging
import shutil
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from .models import TrainingJob, TrainingLog, ModelCheckpoint

logger = logging.getLogger('ml_training')

class DjangoTrainingEngine:
    """Training engine that connects Django admin to actual YOLO training"""
    
    def __init__(self):
        self.base_dir = Path(settings.BASE_DIR)
        
    def start_training_job(self, training_job_id):
        """Start actual YOLO training for a Django training job"""
        
        try:
            # Get training job from database
            job = TrainingJob.objects.get(id=training_job_id)
            
            # Update job status
            job.status = 'running'
            job.started_at = timezone.now()
            job.save()
            
            # Log start
            TrainingLog.objects.create(
                training_job=job,
                level='INFO',
                message=f'Starting YOLO training: {job.name}',
                epoch=0
            )
            
            # Prepare training command
            training_cmd = self.build_training_command(job)
            
            # Execute training
            result = self.execute_training(job, training_cmd)
            
            if result:
                job.status = 'completed'
                job.completed_at = timezone.now()
                TrainingLog.objects.create(
                    training_job=job,
                    level='INFO',
                    message='Training completed successfully',
                    epoch=job.epochs
                )
            else:
                job.status = 'failed'
                TrainingLog.objects.create(
                    training_job=job,
                    level='ERROR',
                    message='Training failed',
                    epoch=job.current_epoch
                )
            
            job.save()
            return result
            
        except Exception as e:
            logger.error(f"Training job {training_job_id} failed: {e}")
            job.status = 'failed'
            job.save()
            return False
    
    def build_training_command(self, job):
        """Build YOLO training command from Django job parameters"""
        
        # YOLO training script (similar to your original training)
        yolo_script = f"""
import os
from ultralytics import YOLO
from pathlib import Path

# Training configuration from Django
model_name = '{job.model_architecture}'
epochs = {job.epochs}
batch_size = {job.batch_size}
learning_rate = {job.learning_rate}
image_size = {job.image_size}

# Load model
if model_name.startswith('yolov8'):
    model = YOLO(f'{{model_name}}.pt')
else:
    model = YOLO('yolov8n.pt')  # Default

# Training parameters
train_args = {{
    'data': '{job.dataset.dataset_path}',  # Path to dataset YAML
    'epochs': epochs,
    'batch': batch_size,
    'lr0': learning_rate,
    'imgsz': {job.image_size},
    'project': '{job.output_path}',
    'name': '{job.name}',
    'exist_ok': True,
    'verbose': True,
    'device': 'cpu'  # Use 'cuda' if GPU available
}}

# Start training
print(f"Starting YOLO training: {{model_name}}")
print(f"Dataset: {{train_args['data']}}")
print(f"Epochs: {{epochs}}, Batch: {{batch_size}}")

results = model.train(**train_args)

print("Training completed!")
print(f"Results saved to: {{results.save_dir}}")
"""
        
        return yolo_script
    
    def execute_training(self, job, training_script):
        """Execute the training script"""
        
        try:
            # Create training directory
            training_dir = Path(job.output_path)
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Write training script
            script_file = training_dir / "train_script.py"
            script_file.write_text(training_script)
            
            # Execute training script
            cmd = [sys.executable, str(script_file)]
            
            # Run training process
            process = subprocess.Popen(
                cmd,
                cwd=training_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor training progress
            for line in process.stdout:
                # Log training output
                TrainingLog.objects.create(
                    training_job=job,
                    level='INFO',
                    message=line.strip(),
                    epoch=self.extract_epoch_from_log(line)
                )
                
                # Update job progress
                epoch = self.extract_epoch_from_log(line)
                if epoch > 0:
                    job.current_epoch = epoch
                    job.progress_percentage = (epoch / job.epochs) * 100
                    job.save()
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                # Training succeeded - save model
                self.save_trained_model(job, training_dir)
                return True
            else:
                logger.error(f"Training failed with return code: {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            return False
    
    def extract_epoch_from_log(self, log_line):
        """Extract epoch number from YOLO training log"""
        try:
            if "Epoch" in log_line and "/" in log_line:
                # Example: "Epoch 5/50"
                parts = log_line.split()
                for part in parts:
                    if "/" in part and part.split("/")[0].isdigit():
                        return int(part.split("/")[0])
            return 0
        except:
            return 0
    
    def save_trained_model(self, job, training_dir):
        """Save the trained model to Django model management"""
        
        try:
            # Find the best model file
            weights_dir = training_dir / job.name / "weights"
            best_model = weights_dir / "best.pt"
            
            if best_model.exists():
                # Create model checkpoint record
                ModelCheckpoint.objects.create(
                    training_job=job,
                    epoch=job.epochs,
                    checkpoint_path=str(best_model),
                    file_size_mb=best_model.stat().st_size / (1024 * 1024),
                    is_best=True,
                    is_latest=True
                )
                
                # Copy to main models directory
                models_root = getattr(settings, 'ML_MODELS_ROOT', Path(settings.BASE_DIR) / 'ml_models')
                dest_model = Path(models_root) / "pest_detection" / f"{job.name}_best.pt"
                dest_model.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(best_model, dest_model)
                
                logger.info(f"Model saved: {dest_model}")
                
                # Create DetectionModel record for use in detection
                from pest_detection.models import DetectionModel
                DetectionModel.objects.create(
                    name=f"{job.name} - Trained Model",
                    model_type='yolo',
                    version='1.0',
                    status='completed',
                    model_path=str(dest_model),
                    training_dataset=job.dataset.name,
                    training_epochs=job.epochs,
                    batch_size=job.batch_size,
                    learning_rate=job.learning_rate,
                    created_by=job.created_by
                )
                
                return True
        except Exception as e:
            logger.error(f"Failed to save trained model: {e}")
            return False
