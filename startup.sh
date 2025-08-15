#!/bin/bash
set -e

echo ">> Starting Medical Assistant API..."

# Use Azure-provided port
: "${PORT:=8000}"

# Use Gunicorn with Uvicorn workers for production
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app \
  --bind 0.0.0.0:"$PORT" \
  --timeout 120
