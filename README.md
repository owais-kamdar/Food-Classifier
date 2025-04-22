# Food Classification System

A multi-approach food classification system using computer vision. This project implements three different approaches to classify food images into 121 different categories, focusing on accuracy and practicality. The system is useful in real-world scenarios such as:
- Identifying unfamiliar dishes at restaurants
- Understanding foreign cuisine while traveling
- Learning about new food items at markets or food festivals
- Getting nutritional information about unknown dishes

## Problem Statement & Overview

Existing challenges:
- High intra-class variance (same food can look very different)
- Inter-class similarity (different foods can look similar)
- Complex backgrounds and lighting conditions
- Cultural and regional variations in food presentation


This project approaches this problem through three approaches:
1. **Naive Classifier**: Feature-based matching
2. **Traditional ML**: Traditional approach using Random Forest Classifier
3. **Deep Learning**: EfficientNet-B0 model

Added Feature:
1. **LLM Based Nutrition and Feedback**: API Call to GPT-3.5-Turbo to provide information for classified food

## [Deployed App](https://food-classifier-w7wtvmoi4nwkrahdh4tjyh.streamlit.app/)


## Previous Efforts in AI-Powered Food Classification and Tracking

### [Food/Non-food Image Classification and Categorization Using GoogLeNet](https://dl.acm.org/doi/abs/10.1145/2986035.2986039)

**Authors**: Ashutosh Singla, Lin Yuan, Touradj Ebrahimi

Two-stage pipeline with a GoogLeNet architecture. First, it filters between food vs. non-food images. In this first stage, researchers were able to achieve an accuracy of 99.2%. After this first fiter, it classifies food images into categories. This also achieved strong performance at 83.6% through minimal fine-tuning, showcasing feature transferability.


### [Personalized Classifier for Food Image Recognition](https://arxiv.org/abs/1804.04600)

**Authors**: Shota Horiguchi, Sosuke Amano, Makoto Ogawa, Kiyoharu Aizawa

Researchers proposed an adaptive model combining 1-Nearest Neighbor and Nearest Class Mean classifiers that addressess the challenge of recognition in real world environments. By using this framework, the classifiers are able to adapt to individual users with minimal samples. Illustrates effectiveness in real-world food logging environments, as users add new food images.


### [Food Classifier Model](https://github.com/darkangrycoder/food_classifier_model)

Uses transfer learning with multiple pre-trained architectures like VGG16, Inception, ResNet to classify different foods. Showed improved accuracy for food classification with few training images by fine-tuning on food-specific datasets. Features a user-friendly interface and nutrition information.


### [Cal AI – AI-Powered Calorie Tracking App](https://www.calai.app/)

A mobile calorie tracking app that uses AI to estimate calories from photos. It reports ~90% accuracy in calorie prediction by using vision-based models trained on large annotated food image datasets. Allows users to snap an image of their meal, and get calculated calories and nutritional information.


### [MyFitnessPal](https://www.myfitnesspal.com/)

Commonly used food and nutrition tracker. Primarily relies on manual input and barcode scanning, but includes a basic camera-based logging feature called Meal Scan. Lacks  true image classification or AI-driven insights compared to other solutions like Cal AI.




## Data Sources

### Datasets
1. [**Food-101 Dataset**](https://www.kaggle.com/datasets/kmader/food41/data)
   - 101 food classes
   - 1,000 images per class
   - Standard benchmark for food classification
   - License: Data files © Original Authors

2. [**Harish Kumar's Food Classification Dataset**](https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset/data)
   - 35 additional food classes
   - 24k unique images
   - Focus on Indian and Asian cuisines
   - CC0: Public Domain


### Combined Dataset
- Total: 126 food classes (merging common classes)
- ~500 images per class
- Balanced representation across cuisines

## Project Structure
```
Food-Estimator/
├── data/
│   ├── raw/                     # Original downloaded datasets
│   ├── processed/               # Processed and augmented images
│   └── final/                   # Final combined dataset
├── models/                      # Trained model files
│   ├── rf_model.pkl             # Traditional ML model
│   ├── efficientnet_b0.pth      # Deep learning model
│   └── features.pkl             # Extracted features
├── scripts/
│   ├── dataset/                 # Data processing scripts
│   ├── naive/                   # Naive scripts
│   ├── trad/                    # Traditional ML scripts
│   ├── deep/                    # Deep learning scripts
│   └── api/                     # API (ChatGPT) scripts
├── notebooks/                   # EDA notebooks
├── app.py                       # Streamlit application
├── requirements.txt             # Project dependencies
├── setup.sh                     # Setup script
└── README.md                  
```

## Data Processing Pipeline

### 1. Data Collection & Organization
```python
# scripts/dataset/download.py
```
- Downloads Food-101 and Harish Kumar datasets
- Organizes into class-specific directories
- EDA on raw data


### 2. Data Combination & Normalization
```python
# scripts/dataset/combine.py
```
- Combines Food-101 (101 classes) with Harish Kumar's dataset (34 classes)
- Normalizes class names by:
  - Converting to lowercase
  - Replacing underscores with spaces
  - Merges duplicate classes
  - Handles similar classes through mappings:
    - 'fries' → 'french fries'
    - 'burger' → 'hamburger'
    - 'donut' → 'donuts'
    - 'taco' → 'tacos'
- Creates unified directory


### 3. Data Augmentation & Splitting
```python
# scripts/dataset/augment.py
```
- Performs image augmentation to balance classes:
  - Random horizontal/vertical flips
  - Random rotations (±10 degrees)
  - Random cropping (85-100% of original size)
- 500 images per class through:
  - Downsampling for classes with >500 images
  - Augmentation for classes with <500 images
- Splits data into:
  - Training: 70% of images
  - Validation: 15% of images
  - Testing: 15% of images


### 4. Feature Extraction & Standardization
```python
# scripts/dataset/make_features.py
```
- Extracts features:
  - HOG features
    - 9 orientations
    - 8x8 pixels per cell
    - 2x2 cells per block
    - L2-Hys normalization
  - Color histogram in HSV space (8x8x8 bins)
  - Edge density using Canny edge detection
- Standardizes features:
  - Converts BGR to RGB color space
  - Scales edge density to [0,1] range
- Saves features in two formats:
  - JSON for naive classifier
  - PKL for traditional ML

## Model Implementation

### 1. Naive Classifier
- **Architecture**:
  - Feature-based matching using average class features
  - Features extracted:
    - HOG features (histogram of oriented gradients)
    - Color histogram in HSV space
    - Edge density
  - L2 distance for nearest neighbor matching
- **Training Process**:
  - Average feature vectors per class
  - Feature matching



### 2. Traditional ML (Random Forest)
- **Architecture**:
  - Random Forest Classifier
  - Preprocessing pipeline:
  - StandardScaler normalization
  - PCA (95% variance retained)
  - Model parameters:
    - 300 estimators
    - Max depth of 35
- **Training Process**:
  - Uses precomputed features from .pkl file
  - Standard scaler for normalization
  - Reduces dimensions with PCA



### 3. Deep Learning (EfficientNet-B0)
- **Architecture**:
  - EfficientNet-B0 base model
  - Transfer learning from ImageNet
  - Custom classification head for food classes
  - Data augmentation pipeline:
    - Random resized crop
    - Random horizontal flip
    - Random rotation (±15°)
    - Color jittering
- **Training Process**:
  - Loads pre-trained EfficientNet-B0
  - Replaces final layer for food classification
  - Uses Adam optimizer (lr=1e-4)
  - Epochs = 15
  - Cross-entropy loss



## Evaluation Metrics
All models were evaluated on test set.

The system uses two primary metrics for evaluation:

1. **Accuracy (Primary Metric)**:
   - Overall classification accuracy
   - Easy to interpret

2. **F1-Score (Secondary Metric)**:
   - Macro average F1-score across all classes
   - Balances precision and recall


## Performance Comparison
| Model | Accuracy | F1-Score | 
|-------|----------|----------|
| Naive | 0.0585   | 0.05     | 
| RF    | 0.3122   | 0.33     | 
| Deep  | 0.8007   | 0.80     |


## Installation
Setup script to handle installation and environment:

```bash
# Run the setup script
./setup.sh
```

The setup script performs the following steps:
1. **Environment Setup**:
   - Creates and activates a Python virtual environment
   - Installs required dependencies from requirements.txt

2. **Data Preparation**:
   - Downloads Food-101 and Harish Kumar datasets
   - Combines datasets into a unified structure
   - Performs image augmentation
   - Extracts features for all models

3. **Model Training**:
   - Trains the Random Forest model
   - Deep Model was trained on Google Colab with GPU



## Usage
1. **Run the Application Locally**:
   ```bash
   streamlit run app.py
   ```

2. **Add OpenAI Key to .env file**

2. **Model Selection**:
   - Upload food images for classification
   - View predictions and nutritional information


## Challenges
1. **Merging Dataset**:
   - Varying class names and categories

2. **Class Imbalance**:
   - Different food classes have varying numbers of images

3. **Feature Extraction**:
   - Food images have high variance

4. **Computational Resources**:
   - Large dataset and Feature files (121 classes × 500 images)
   - Time-consuming training even for Traditional ML

5. **API Name Matching Limitations**:
   - The detected food wasn't properly or accurately outputed through several food APIs



## [Presentation Slides](https://docs.google.com/presentation/d/1RSxAnuA1UHcGn0WJa4XNEhkY_5KclEkmPBDFUnOo4aE/edit?usp=sharing)

## [Hosted Models](https://huggingface.co/okamdar/food-classification/tree/main)


## Ethics Statement
1. **Privacy**: No collection of personal data of identifying information
2. **Accessibility**: Open-source implementations and clear documentation
3. **Responsible Use**: Licensed under MIT and intended for educational and research purposes


### ChatGPT-4o and Cursor were used to produce several scripts in this codebase on 4/20/2025. Goal was to reorganize code notebooks into organized, modular, and polished scripts. Several prompts were used such as "organize this code into a modular and cohesive script" etc. 
