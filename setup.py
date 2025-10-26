#!/usr/bin/env python3
"""
Simple Django Setup Script
"""

import os
import django
from pathlib import Path

def main():
    print("AgriShield ML Django Setup")
    
    # Set Django settings
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agrihield_django.settings')
    django.setup()
    
    # Run migrations
    print("Running migrations...")
    os.system("python manage.py makemigrations")
    os.system("python manage.py migrate")
    
    # Create superuser (interactive)
    print("Create superuser account:")
    os.system("python manage.py createsuperuser")
    
    print("Setup complete!")
    print("Start server: python manage.py runserver")

if __name__ == "__main__":
    main()
