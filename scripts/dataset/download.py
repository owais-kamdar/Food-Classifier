"""
Downloads food datasets from Kaggle.
Uses Food-101 and Harish Kumar's food classification dataset.

This script requires:
Kaggle API credentials (kaggle.json) in ~/.kaggle/
"""

import os
import shutil
import subprocess


def download_food101():
    """Download and extract the Food-101 dataset from Kaggle.
    
    This function creates necessary directories, 
    downloads the Food-101 dataset, 
    and extracts the zip file.
    """
    print("Downloading Food-101 dataset...")
    
    # Set up paths for the Food-101 dataset
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    dataset_dir = os.path.join(PROJECT_ROOT, "data", "raw", "food101")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download dataset using Kaggle CLI
    subprocess.run(
        f"kaggle datasets download -d kmader/food41 -p {dataset_dir}".split(),
        check=True
    )
    
    # Extract and clean up the downloaded zip file
    zip_file = os.path.join(dataset_dir, "food41.zip")
    if os.path.exists(zip_file):
        shutil.unpack_archive(zip_file, dataset_dir)
        os.remove(zip_file)
        print("Food-101 dataset downloaded and extracted")
    else:
        print("Error: Food-101 download failed")


def download_food_classification():
    """Download and extract Harish Kumar's food classification dataset from Kaggle.
    
    This function creates necessary directories, 
    downloads the food classification dataset, 
    and extracts the zip file.
    """
    print("Downloading Food Image Classification dataset...")
    
    # Set up paths for the food classification dataset
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    dataset_dir = os.path.join(PROJECT_ROOT, "data", "raw", "food_classification")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download dataset using Kaggle CLI
    subprocess.run(
        f"kaggle datasets download -d harishkumardatalab/food-image-classification-dataset -p {dataset_dir}".split(),
        check=True
    )
    
    # Extract and clean up the downloaded zip file
    zip_file = os.path.join(dataset_dir, "food-image-classification-dataset.zip")
    if os.path.exists(zip_file):
        shutil.unpack_archive(zip_file, dataset_dir)
        os.remove(zip_file)
        print("Food Image Classification dataset downloaded and extracted")
    else:
        print("Error: Food Image Classification download failed")


def main():
    """Main function to download all required food datasets.
    
    This function:
    1. Downloads the Food-101 dataset
    2. Downloads Harish Kumar's food classification dataset
    
    Both datasets will be saved in their respective directories under data/raw/
    """
    download_food101()
    download_food_classification()


if __name__ == "__main__":
    main()