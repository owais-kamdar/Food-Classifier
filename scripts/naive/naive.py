"""
Implements a naive classifier for food image classification.
This classifier uses simple feature extraction and nearest neighbor matching.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.append(project_root)

from scripts.api.food_summary import get_food_summary
from scripts.dataset.make_features import extract_features
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import json

def classify_image_by_features(uploaded_image, feature_index):
    """Predict food class based on image features and precomputed feature index.
    
    This function: Decodes the uploaded image, 
    Extracts features from the image, 
    and finds the closest matching class
    
    Args:
        uploaded_image: Uploaded image file object
        feature_index (dict): Dictionary mapping class names to their feature vectors
        
    Returns:
        str: Name of the predicted food class
        str: Error message if image decoding fails
    """
    # Decode the uploaded image
    image_array = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    if image_array is None:
        return None, "Failed to decode image."
    
    # Extract features from the image
    query_features = extract_features(image_array)

    # Find closest match using Euclidean distance
    best_class, best_dist = None, float("inf")
    for class_name, vec in feature_index.items():
        dist = np.linalg.norm(query_features - vec)
        if dist < best_dist:
            best_class, best_dist = class_name, dist

    return best_class

# Evaluate the naive classifier on test data
def evaluate_naive_classifier(test_dir, feature_index):
    """Evaluate the naive classifier on test data.
    This function:
    1. Calculates accuracy
    2. Generates classification report
    3. Creates and saves confusion matrix
    
    Args:
        test_dir (str): Directory containing test images
        feature_index (dict): Dictionary mapping class names to their feature vectors
        
    Returns:
        tuple: (accuracy, classification report, confusion matrix)
    """
    true_labels = []
    pred_labels = []
    class_names = sorted(os.listdir(test_dir))
    
    # Iterate through each class directory
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Process each image in the class directory
        for img_name in tqdm(os.listdir(class_dir), desc=f"Evaluating {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            
            # Skip if it's a directory
            if os.path.isdir(img_path):
                continue
                
            # Read and predict
            with open(img_path, 'rb') as f:
                pred_class = classify_image_by_features(f, feature_index)
            
            true_labels.append(class_name)
            pred_labels.append(pred_class)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Generate classification report
    report = classification_report(true_labels, pred_labels, zero_division=0)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    plt.figure(figsize=(20, 20))
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
    print(f"Naive Classifier Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print(f"Confusion matrix saved to: confusion_matrix.png")
    
    return accuracy, report, cm

if __name__ == "__main__":
    # Load feature index
    feature_index_path = os.path.join(project_root, "scripts/naive/feature_index.json")
    with open(feature_index_path, "r") as f:
        feature_index = json.load(f)
    
    # Evaluate on test set
    test_dir = os.path.join(project_root, "data/final/splits/test/images")
    evaluate_naive_classifier(test_dir, feature_index)
    print("Naive Classifier Evaluation Complete")
