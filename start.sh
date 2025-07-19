#!/bin/bash

# LogPrompt API Startup Script

echo "🚀 Starting LogPrompt API..."

# Check if pipenv is available
if command -v pipenv &> /dev/null; then
    echo "📦 Using Pipenv..."
    pipenv install
    pipenv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "🐍 Using pip and venv..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3.12 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    pip install -r requirements.txt
    
    # Run the application
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
fi
