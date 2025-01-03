#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running migrations for auth and contenttypes..."
python manage.py migrate auth
python manage.py migrate contenttypes

echo "Running all other migrations..."
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --no-input --clear

echo "Creating superuser..."
DJANGO_SUPERUSER_PASSWORD=${DJANGO_SUPERUSER_PASSWORD} python manage.py createsuperuser --noinput \
    --username ${DJANGO_SUPERUSER_USERNAME} \
    --email ${DJANGO_SUPERUSER_EMAIL} || true 