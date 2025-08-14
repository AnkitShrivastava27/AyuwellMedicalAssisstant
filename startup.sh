#!/bin/bash

# Exit on error
set -e

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Start FastAPI with Uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8000
