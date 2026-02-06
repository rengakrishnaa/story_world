#!/bin/bash
# Replit: use PORT from Cloud Run (default 8080) or 8000 for local Run
PORT=${PORT:-8000}
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
