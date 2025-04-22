import os
import cv2
import json
import joblib
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from scipy.stats import entropy

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

# Path to the training images
TRAIN_DIR = os.path.join(project_root, "data/final/splits/train/images")
NAIVE_OUTPUT_PATH = os.path.join(project_root, "scripts/naive/feature_index.json")
TRAD_OUTPUT_PATH = os.path.join(project_root, "models/features.pkl")

def extract_features(image, resize=(128, 128)):
    """
    Extract essential image features for food recognition:
    - HOG features (histogram of oriented gradients) for texture and shape
    - Color histogram in HSV space for color information
    - Edge density for texture emphasis
    
    Args:
        image: numpy array of the image
        resize: tuple of (width, height) to resize image to
        
    Returns:
        numpy array of concatenated features
    """
    # Resize image for consistent feature extraction
    image = cv2.resize(image, resize)
    features = []

    # Convert to RGB if image is in BGR format
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. HOG Features
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    features.extend(hog_features)

    # 2. Color Histogram (HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    features.extend(hist)

    # 3. Edge Density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255.0)
    features.append(edge_density)

    return np.array(features)

def build_feature_index():
    """
    Build feature index for naive and traditional ML classifiers.
    Saves both JSON format for naive classifier and PKL format for traditional ML.
    """
    index = {}  # For naive classifier (JSON)
    X = []      # For traditional ML (PKL)
    y = []      # For traditional ML (PKL)

    # Get directories
    food_dirs = [d for d in os.listdir(TRAIN_DIR) 
                if os.path.isdir(os.path.join(TRAIN_DIR, d)) 
                and not d.startswith('.')]

    # Extract features for each food class
    for class_name in tqdm(food_dirs, desc="Extracting features"):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        feature_list = []

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # Read image
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipped unreadable image: {img_path}")
                continue

            # Extract features
            features = extract_features(image)
            feature_list.append(features)

            # For traditional ML, save all individual features
            X.append(features)
            y.append(class_name)

        if feature_list:
            # For naive classifier, compute average features for this class
            class_avg_features = np.mean(feature_list, axis=0).tolist()
            index[class_name] = class_avg_features
        else:
            print(f"No valid images found in class '{class_name}'")

    # Save naive classifier features (JSON)
    with open(NAIVE_OUTPUT_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Naive feature index saved to: {NAIVE_OUTPUT_PATH} ({len(index)} classes)")

    # Save traditional ML features (PKL)
    joblib.dump((X, y), TRAD_OUTPUT_PATH)
    print(f"Traditional ML features saved to: {TRAD_OUTPUT_PATH} ({len(X)} images)")

if __name__ == "__main__":
    build_feature_index() 