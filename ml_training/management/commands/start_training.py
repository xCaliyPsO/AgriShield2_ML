from django.core.management.base import BaseCommand
from ml_training.models import TrainingJob
from ml_training.training_engine import DjangoTrainingEngine

class Command(BaseCommand):
    help = 'Start a training job'
    
    def add_arguments(self, parser):
        parser.add_argument('job_id', type=int, help='Training job ID')
    
    def handle(self, *args, **options):
        job_id = options['job_id']
        
        try:
            job = TrainingJob.objects.get(id=job_id)
            self.stdout.write(f'Starting training job: {job.name}')
            
            engine = DjangoTrainingEngine()
            success = engine.start_training_job(job_id)
            
            if success:
                self.stdout.write(
                    self.style.SUCCESS(f'Training job {job_id} completed successfully')
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'Training job {job_id} failed')
                )
                
        except TrainingJob.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'Training job {job_id} not found')
            )
