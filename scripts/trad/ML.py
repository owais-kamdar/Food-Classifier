"""
Implements a traditional ML classifier for food image classification.
This classifier uses Random Forest with feature extraction and dimensionality reduction.
"""

import os
import cv2
import json
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
from joblib import Parallel
from joblib import delayed
from huggingface_hub import hf_hub_download

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.append(project_root)

from scripts.dataset.make_features import extract_features

def train_model():
    """Train a Random Forest model for food classification.
    This function:
    1. Loads features
    2. Creates a pipeline with feature scaling and PCA
    3. Trains and saves a Random Forest classifier
    
    Returns:
        sklearn.pipeline.Pipeline: Trained model
    """
    # Load pre-calculated features from make_features.py
    features_path = os.path.join(project_root, "models/features.pkl")
    X, y = joblib.load(features_path)
    
    # Convert features to numpy array if they're a list
    X = np.array(X)
    
    print(f"Loaded features for {len(X)} training images")
    print(f"Classes: {sorted(set(y))}")
    print(f"Original feature dimension: {X.shape[1]}")

    # Random Forest with feature scaling and PCA
    print("Creating Random Forest model...")
    model = make_pipeline(
        StandardScaler(), 
        PCA(n_components=0.95, random_state=42),
        RandomForestClassifier(
            n_estimators=300,        # Number of trees
            max_depth=35,            
            min_samples_split=5,     # Minimum samples to split
            min_samples_leaf=2,      # Minimum samples in leaf
            max_features='sqrt',
            n_jobs=-1,               # Use all available cores
            verbose=1                # Show progress
        )
    )
    
    print("Training model...")
    # Add time for debugging
    start_time = time.time()
    model.fit(X, y)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Save model
    model_path = os.path.join(project_root, "models", "rf_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model

def evaluate_ml_classifier(test_dir, model):
    """Evaluate the ML classifier on test data.
    This function:
    1. Calculates accuracy
    2. Generates classification report
    3. Creates and saves confusion matrix
    
    Args:
        test_dir (str): Directory containing test images
        model (sklearn.pipeline.Pipeline): Trained Random Forest model
    Returns:
        tuple: (accuracy, classification report, confusion matrix)
    """
    # Get all image paths and labels first
    image_paths = []
    true_labels = []
    class_names = sorted(os.listdir(test_dir))
    
    print("Collecting image paths...")
    for class_name in tqdm(class_names, desc="Collecting paths"):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isdir(img_path):
                image_paths.append(img_path)
                true_labels.append(class_name)
    
    # Process images in batches to speed up processing
    batch_size = 100
    pred_labels = []
    
    print(f"\nProcessing {len(image_paths)} images in batches of {batch_size}...")
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_features = []
        
        # Load and process images in parallel
        with Parallel(n_jobs=11) as parallel:
            batch_features = parallel(
                delayed(lambda p: extract_features(cv2.imread(p)) if cv2.imread(p) is not None else None)(img_path) 
                for img_path in batch_paths
            )
        
        # Remove None values (failed image loads)
        valid_features = [f for f in batch_features if f is not None]
        if valid_features:
            # Make predictions for the batch
            batch_predictions = model.predict(valid_features)
            pred_labels.extend(batch_predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels[:len(pred_labels)], pred_labels)
    report = classification_report(true_labels[:len(pred_labels)], pred_labels)
    
    # Generate and display confusion matrix
    cm = confusion_matrix(true_labels[:len(pred_labels)], pred_labels, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    # Print evaluation results
    print(f"\nRandom Forest Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix saved to: confusion_matrix.png")
    
    return accuracy, report, cm

def predict_from_image(image_array):
    """Predict food class from an image array using the trained Random Forest model.
    This function: Downloads and loads the model from Hugging Face, then makes a prediction
    Args:
        image_array (numpy.ndarray): Input image as numpy array
    Returns:
        str: Predicted food class name
    Raises:
        ValueError: If the image cannot be decoded
    """
    # Download model from Hugging Face
    model_path = hf_hub_download(
        repo_id="okamdar/food-classification",
        filename="rf_model.pkl",
        local_dir=os.path.join(project_root, "models")
    )
    
    # Load the model
    model = joblib.load(model_path)
    
    if image_array is None:
        raise ValueError("Cannot decode image")

    # Extract features and make prediction
    features = extract_features(image_array).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

if __name__ == "__main__":
    # Train and save the model
    model = train_model()
    
    # Evaluate on test set
    test_dir = os.path.join(project_root, "data/final/splits/test/images")
    evaluate_ml_classifier(test_dir, model)
    print("Random Forest Classifier Training and Evaluation Complete")