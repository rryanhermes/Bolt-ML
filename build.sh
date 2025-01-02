#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --no-input

echo "Creating superuser..."
python manage.py createsuperuser --noinput --username admin --email admin@example.com || true 