#!/bin/bash

# Activate virtual environment if using one
# source .venv/bin/activate

# Make sure the required directories exist
mkdir -p uploads
mkdir -p models
mkdir -p data

# Run the Flask server
python app.py 