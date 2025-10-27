from django.core.management.base import BaseCommand
from pest_detection.models import PestType, DetectionModel


class Command(BaseCommand):
    help = 'Populate initial pest data and create default detection model'

    def handle(self, *args, **options):
        self.stdout.write('Creating initial pest types...')
        
        # Create initial pest types with their confidence thresholds
        pest_data = [
            {
                'name': 'Rice_Bug',
                'scientific_name': 'Leptocorisa oratorius',
                'description': 'Rice bug that feeds on rice grains causing damage',
                'damage_description': 'Feeds on developing rice grains, causing empty or malformed grains',
                'diseases_caused': ['Grain damage', 'Yield reduction'],
                'pesticide_recommendations': 'Use appropriate insecticides during flowering stage',
                'biological_control': 'Natural predators like spiders and birds',
                'cultural_control': 'Proper field sanitation and crop rotation',
                'confidence_threshold': 0.20
            },
            {
                'name': 'black-bug',
                'scientific_name': 'Scotinophara coarctata',
                'description': 'Black rice bug that causes significant damage',
                'damage_description': 'Sucks sap from rice plants causing yellowing and stunting',
                'diseases_caused': ['Plant stunting', 'Yellowing'],
                'pesticide_recommendations': 'Systemic insecticides recommended',
                'biological_control': 'Parasitic wasps and predatory beetles',
                'cultural_control': 'Early planting and resistant varieties',
                'confidence_threshold': 0.80
            },
            {
                'name': 'brown_hopper',
                'scientific_name': 'Nilaparvata lugens',
                'description': 'Brown planthopper that transmits rice diseases',
                'damage_description': 'Feeds on rice plants and transmits viral diseases',
                'diseases_caused': ['Rice ragged stunt virus', 'Rice grassy stunt virus'],
                'pesticide_recommendations': 'Avoid excessive nitrogen, use selective insecticides',
                'biological_control': 'Spiders, mirid bugs, and parasitic wasps',
                'cultural_control': 'Resistant varieties and proper water management',
                'confidence_threshold': 0.15
            },
            {
                'name': 'green_hopper',
                'scientific_name': 'Nephotettix virescens',
                'description': 'Green leafhopper that transmits rice diseases',
                'damage_description': 'Feeds on rice leaves and transmits tungro virus',
                'diseases_caused': ['Rice tungro virus', 'Yellow dwarf virus'],
                'pesticide_recommendations': 'Systemic insecticides during early stages',
                'biological_control': 'Predatory spiders and parasitic wasps',
                'cultural_control': 'Resistant varieties and synchronized planting',
                'confidence_threshold': 0.15
            }
        ]
        
        created_count = 0
        for pest_info in pest_data:
            pest, created = PestType.objects.get_or_create(
                name=pest_info['name'],
                defaults=pest_info
            )
            if created:
                created_count += 1
                self.stdout.write(f'  Created: {pest.name}')
            else:
                self.stdout.write(f'  Already exists: {pest.name}')
        
        self.stdout.write(f'Created {created_count} new pest types')
        
        # Create default detection model if it doesn't exist
        self.stdout.write('Creating default detection model...')
        
        default_model, created = DetectionModel.objects.get_or_create(
            name='Default Pest Detection Model',
            defaults={
                'model_type': 'yolo',
                'version': '1.0',
                'status': 'completed',
                'model_path': 'ml_models/pest_detection/best.pt',
                'training_dataset': 'Default Dataset',
                'training_epochs': 50,
                'batch_size': 8,
                'learning_rate': 0.001,
                'is_active': True
            }
        )
        
        if created:
            # Link all pest types to the default model
            default_model.pest_types.set(PestType.objects.all())
            self.stdout.write('  Created default detection model and linked pest types')
        else:
            self.stdout.write('  Default detection model already exists')
        
        self.stdout.write(
            self.style.SUCCESS('Successfully populated initial pest data!')
        )
