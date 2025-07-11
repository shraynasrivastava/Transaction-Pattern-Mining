#!/bin/bash

# Exit on error
set -e

echo "Setting up Python virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading dataset..."
python scripts/download_data.py

echo "Running tests..."
python tests/test_pipeline.py

echo "Running main analysis..."
python src/main.py --optimize-clusters

echo "Analysis complete! Check the data/processed/visualizations directory for results." 