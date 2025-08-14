#!/bin/bash

# Exit immediately if any command fails
set -e

# Install dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Run FastAPI app with Gunicorn + Uvicorn workers for production
echo "Starting Medical Assistant API on Azure..."
gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind=0.0.0.0:${PORT:-8000}
