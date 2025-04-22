#!/bin/bash

# Exit on error
set -e

echo " Setting up Food Classifier Project..."

# Clone repository if not already cloned
if [ ! -d "Food-Estimator" ]; then
    echo "Cloning repository..."
    git clone https://github.com/yourusername/Food-Estimator.git
    cd Food-Estimator
else
    echo "Repository already exists, skipping clone..."
    cd Food-Estimator
fi

# Create and activate virtual environment
echo " Setting up virtual environment..."
python -m venv venv
source venv/bin/activate

# Install requirements
echo " Installing requirements..."
pip install -r requirements.txt

# Download datasets
echo " Downloading datasets..."
python scripts/dataset/download.py

# Combine datasets
echo " Combining datasets..."
python scripts/dataset/combine.py

# Augment images
echo " Augmenting images..."
python scripts/dataset/augment.py

# Extract features
echo " Extracting features..."
python scripts/dataset/make_features.py

# Train ML model
echo " Training ML model..."
python scripts/trad/ML.py
