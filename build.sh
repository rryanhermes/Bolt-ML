#!/bin/bash

# Wait for the database to be ready
echo "Waiting for database..."
sleep 10

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Start Gunicorn
echo "Starting Gunicorn..."
exec gunicorn --bind :$PORT myproject.wsgi:application 