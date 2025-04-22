"""
Augments food images and splits them into train/val/test sets.
Adds flipped, rotated, and brightness-adjusted versions of images to balance classes.
"""

import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import random

# Paths and settings
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
target_per_class = 500  # Target number of images per food class
valid_split = 0.15      # 15% for validation
test_split = 0.15       # 15% for testing


def augment_image(image):
    """Create more images through key geometric transformations, preserving original colors.

    Args: 
        image: Input image to augment
    Returns: 
        list: List of augmented images
    """
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    augmented = []
    h, w = image.shape[:2]
    
    # Horizontal flip (most natural for food)
    augmented.append(cv2.flip(image, 1))
    
    # Vertical flip
    augmented.append(cv2.flip(image, 0))
    
    # Small random rotations (±10 degrees)
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)
    
    # Random crop with resizing (to handle different framing)
    crop_size = random.uniform(0.85, 1.0)  # Crop between 85-100% of original size
    new_h, new_w = int(h * crop_size), int(w * crop_size)
    y = random.randint(0, h - new_h)
    x = random.randint(0, w - new_w)
    cropped = image[y:y+new_h, x:x+new_w]
    resized = cv2.resize(cropped, (w, h))
    augmented.append(resized)
    
    return augmented


def split_and_copy(image_paths, class_name, final_dir):
    """Split images into train/val/test sets and copy them to the right places.
    
    Args:
        image_paths (list): List of paths to images
        class_name (str): Name of the food
        final_dir (str): Directory for the final split dataset
    """
    # Split into train/val/test
    trainval, test = train_test_split(image_paths, test_size=test_split, random_state=42)
    train, val = train_test_split(trainval, test_size=valid_split / (1 - test_split), random_state=42)

    # Copy files to their new directories
    for split, files in zip(['train', 'val', 'test'], [train, val, test]):
        out_dir = os.path.join(final_dir, split, "images", class_name)
        os.makedirs(out_dir, exist_ok=True)
        for path in files:
            try:
                shutil.copy(path, os.path.join(out_dir, os.path.basename(path)))
            except Exception as e:
                print(f"Copy failed for {path}: {e}")


def process_class(class_dir, aug_dir, final_dir):
    """Process all images in a food class directory.
    
    This function creates augmented versions of images if needed, 
    balances the number of images per class, 
    splits images into train/val/test sets.
    
    Args:
        class_dir (str): Directory for original images
        aug_dir (str): Directory for augmented images
        final_dir (str): Directory for the final split dataset
    """
    class_name = os.path.basename(class_dir)
    
    # Create augmented directory for this class
    aug_class_dir = os.path.join(aug_dir, class_name)
    os.makedirs(aug_class_dir, exist_ok=True)
    
    # Get all original images
    original_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nClass: {class_name}")
    print(f"Original images: {len(original_images)}")

    # If we have too many images, randomly sample down to target
    if len(original_images) > target_per_class:
        print(f"Downsampling from {len(original_images)} to {target_per_class} images")
        original_images = random.sample(original_images, target_per_class)
    
    # Copy original images to augmented directory
    for img_path in original_images:
        shutil.copy(img_path, os.path.join(aug_class_dir, os.path.basename(img_path)))

    # If we don't have enough images, create augmented versions
    if len(original_images) < target_per_class:
        need = target_per_class - len(original_images)
        print(f"Need {need} more images through augmentation")
        
        # Create augmented versions until we reach target
        aug_count = 0
        while aug_count < need:
            for img_path in original_images:
                if aug_count >= need:
                    break
                    
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    # Get augmented versions
                    aug_images = augment_image(img)
                    
                    # Save each augmented version
                    for aug_idx, aug_img in enumerate(aug_images):
                        if aug_count >= need:
                            break
                            
                        aug_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{aug_count}.jpg"
                        aug_path = os.path.join(aug_class_dir, aug_name)
                        cv2.imwrite(aug_path, aug_img)
                        aug_count += 1
                        
                except Exception as e:
                    print(f"Error augmenting {img_path}: {e}")

        print(f"Added {aug_count} augmented images")

    # Get all images in augmented directory (original + augmented)
    all_images = [os.path.join(aug_class_dir, f) for f in os.listdir(aug_class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total images after processing: {len(all_images)}")

    # Split up our images and copy them to their final destinations
    split_and_copy(all_images, class_name, final_dir)


if __name__ == "__main__":
    # Set up our directories
    source_dir = os.path.join(project_root, "data/processed/combined")
    aug_dir = os.path.join(project_root, "data/processed/augmented")
    final_dir = os.path.join(project_root, "data/final/splits")
    
    # Create augmented directory
    os.makedirs(aug_dir, exist_ok=True)
    
    # Process each food class
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        process_class(class_dir, aug_dir, final_dir)
