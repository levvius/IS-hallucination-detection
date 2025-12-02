#!/bin/bash

# Script to run the Text Classification API

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Text Classification API Setup ==="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Check if Knowledge Base exists
if [ ! -f "data/faiss_index/wikipedia.index" ]; then
    echo ""
    echo "⚠️  Knowledge Base not found!"
    echo "Building Knowledge Base from Wikipedia..."
    echo "This will take 5-10 minutes..."
    echo ""
    python scripts/build_kb.py
fi

# Run the API server
echo ""
echo "=== Starting API Server ==="
echo "API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo "Health check at: http://localhost:8000/api/v1/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
