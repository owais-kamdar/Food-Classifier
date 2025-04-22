"""
Combines multiple food datasets into one directory.
Merges Food-101 and Harish Kumar's dataset into a single processed folder.
"""

import os
import shutil
from pathlib import Path
import sys

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

# Define class name mappings to match Food-101 class names
CLASS_MAPPINGS = {
    'fries': 'french fries',
    'burger': 'hamburger',
    'donut': 'donuts',
    'taco': 'tacos'
}

# Normalize class names
def normalize_class_name(name):
    """Normalize class name by converting to lowercase and replacing underscores with spaces.
    
    Args:
        name (str): Original class name
        
    Returns:
        str: Normalized class name
    """
    # Convert to lowercase and replace underscores with spaces
    norm_name = name.lower().replace("_", " ")
    
    # Then check if it should be mapped to another class
    return CLASS_MAPPINGS.get(norm_name, norm_name)

def combine_datasets():
    """Combine all food images into a single processed directory.
    
    This function:
    1. Sets up paths for input and output directories
    2. Processes and combines both datasets into a single processed directory
    3. Normalizes class names and handles duplicates
    """
    # Set up paths for input and output directories
    raw_food101 = os.path.join(project_root, "data/raw/food101/images")
    raw_harish = os.path.join(project_root, "data/raw/food_classification/Food Classification dataset")
    processed_dir = os.path.join(project_root, "data/processed/combined")
    
    # Create output directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Dictionary to track class counts
    class_counts = {}
    
    # Process Food-101 dataset
    print("Processing Food-101 dataset...")
    if os.path.exists(raw_food101):
        for class_name in os.listdir(raw_food101):
            class_dir = os.path.join(raw_food101, class_name)
            if os.path.isdir(class_dir):
                # Normalize class name
                norm_class = normalize_class_name(class_name)
                
                # Create class directory in processed folder
                processed_class_dir = os.path.join(processed_dir, norm_class)
                os.makedirs(processed_class_dir, exist_ok=True)
                
                # Copy images from Food-101 to processed directory
                for image in os.listdir(class_dir):
                    if not image.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    src = os.path.join(class_dir, image)
                    dst = os.path.join(processed_class_dir, image)
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"Error copying {src}: {e}")
                        continue
                
                # Update class count
                class_counts[norm_class] = len(os.listdir(processed_class_dir))
    
    # Process Harish Kumar dataset
    print("Processing Harish Kumar dataset...")
    if os.path.exists(raw_harish):
        for class_name in os.listdir(raw_harish):
            class_dir = os.path.join(raw_harish, class_name)
            if os.path.isdir(class_dir):
                # Normalize class name
                norm_class = normalize_class_name(class_name)
                
                # Create class directory in processed folder
                processed_class_dir = os.path.join(processed_dir, norm_class)
                os.makedirs(processed_class_dir, exist_ok=True)
                
                # Copy images from Harish Kumar dataset to processed directory
                for image in os.listdir(class_dir):
                    if not image.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    src = os.path.join(class_dir, image)
                    dst = os.path.join(processed_class_dir, image)
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"Error copying {src}: {e}")
                        continue
                
                # Update class count
                if norm_class in class_counts:
                    class_counts[norm_class] += len(os.listdir(class_dir))
                else:
                    class_counts[norm_class] = len(os.listdir(class_dir))
    
    # Print statistics about the combined dataset
    total_classes = len(class_counts)
    total_images = sum(class_counts.values())
    
    print("\nDataset combination complete!")
    print(f"Total classes: {total_classes}")
    print(f"Total images: {total_images}")
    print("\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count} images")
    print(f"\nCombined dataset saved to: {processed_dir}")

if __name__ == "__main__":
    combine_datasets()
