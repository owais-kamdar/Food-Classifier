import streamlit as st
import cv2
import numpy as np
import json
import joblib
from PIL import Image
import io
import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Get the correct project root
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(project_root)

# Import classifiers
from scripts.naive.naive import classify_image_by_features
from scripts.api.food_summary import get_food_summary
from scripts.deep.train import predict_from_image as predict_from_image_deep
from scripts.trad.ML import predict_from_image as predict_from_image_ml


# Constants for deep learning
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use listed class names in order

class_names = ['apple pie', 'baby back ribs', 'baked potato', 'baklava', 
               'beef carpaccio', 'beef tartare', 'beet salad', 'beignets', 
               'bibimbap', 'bread pudding', 'breakfast burrito', 'bruschetta', 
               'butter naan', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake', 
               'ceviche', 'chai', 'chapati', 'cheese plate', 'cheesecake', 'chicken curry', 
               'chicken quesadilla', 'chicken wings', 'chocolate cake', 'chocolate mousse', 
               'chole bhature', 'churros', 'clam chowder', 'club sandwich', 'crab cakes', 
               'creme brulee', 'crispy chicken', 'croque madame', 'cup cakes', 'dal makhani', 
               'deviled eggs', 'dhokla', 'donuts', 'dumplings', 'edamame', 'eggs benedict', 
               'escargots', 'falafel', 'filet mignon', 'fish and chips', 'foie gras', 'french fries', 
               'french onion soup', 'french toast', 'fried calamari', 'fried rice', 'frozen yogurt', 
               'garlic bread', 'gnocchi', 'greek salad', 'grilled cheese sandwich', 'grilled salmon', 
               'guacamole', 'gyoza', 'hamburger', 'hot and sour soup', 'hot dog', 'huevos rancheros', 
               'hummus', 'ice cream', 'idli', 'jalebi', 'kaathi rolls', 'kadai paneer', 'kulfi', 
               'lasagna', 'lobster bisque', 'lobster roll sandwich', 'macaroni and cheese', 'macarons', 
               'masala dosa', 'miso soup', 'momos', 'mussels', 'nachos', 'omelette', 'onion rings', 
               'oysters', 'paani puri', 'pad thai', 'paella', 'pakode', 'pancakes', 'panna cotta', 
               'pav bhaji', 'peking duck', 'pho', 'pizza', 'pork chop', 'poutine', 'prime rib', 
               'pulled pork sandwich', 'ramen', 'ravioli', 'red velvet cake', 'risotto', 'samosa', 
               'sandwich', 'sashimi', 'scallops', 'seaweed salad', 'shrimp and grits', 'spaghetti bolognese', 
               'spaghetti carbonara', 'spring rolls', 'steak', 'strawberry shortcake', 'sushi', 'tacos', 
               'takoyaki', 'taquito', 'tiramisu', 'tuna tartare', 'waffles']

# Load necessary data
@st.cache_data
def load_data():
    # Load naive classifier feature index
    feature_index_path = "scripts/naive/feature_index.json"
    if not os.path.exists(feature_index_path):
        st.error(f"Feature index not found at: {feature_index_path}")
        return None
    with open(feature_index_path, "r") as f:
        feature_index = json.load(f)
    return feature_index

def main():
    st.title("🍽️ Food Classifier")
    
    # Add description section
    st.markdown("""
       ### How to Use:
    1. Select a classifier from the sidebar
    2. Upload a food image (JPG, JPEG, or PNG)
    3. View the prediction and nutritional information
    """)
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Selection")
        model_type = st.radio(
            "Choose Classifier:",
            ["Naive Classifier", "Traditional ML", "Deep Learning"],
            index=2  # Default to Deep Learning
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses computer vision to classify food images and provide nutritional information.
        
        Available models:
        - Naive Classifier: Simple Feature-based classification
        - Traditional ML: Random Forest Classification
        - Deep Learning: EfficientNet-B0 model trained on 121 food classes
        """)

    # Load data
    feature_index = load_data()
    if feature_index is None:
        st.error("Failed to load feature index. Please check the file path.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image as numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_array is None:
            st.error("❌ Failed to decode image")
            return

        # Convert BGR to RGB for display and processing
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Display the uploaded image
        st.image(img_array, caption="Uploaded Image", use_container_width=True)

        # Process based on selected model
        if model_type == "Naive Classifier":
            try:
                # Reset file pointer for naive classifier
                uploaded_file.seek(0)
                prediction = classify_image_by_features(uploaded_file, feature_index)
                
                # Display prediction
                st.subheader(f"**Prediction:** {prediction}")
                
                # Get and display food summary
                if prediction:
                    summary = get_food_summary(prediction)
                    st.subheader("**Food Summary:**")
                    st.write(summary)
            except Exception as e:
                st.error(f"Error in naive classifier: {str(e)}")
        elif model_type == "Traditional ML":
            try:
                # Make prediction using Random Forest
                prediction = predict_from_image_ml(img_array)
                
                # Display results
                st.subheader(f"**Prediction:** {prediction}")
                
                # Get and display food summary
                summary = get_food_summary(prediction)
                st.subheader("**Food Summary:**")
                st.write(summary)
            except Exception as e:
                st.error(f"Error in Random Forest classifier: {str(e)}")
        else:  # Deep Learning
            try:
                # Make prediction using the imported function
                prediction, confidence = predict_from_image_deep(img_array)
                
                # Display results
                st.subheader(f"**Prediction:** {prediction}")
                # st.write(f"**Confidence:** {confidence:.2%}")
                
                # Get and display food summary
                summary = get_food_summary(prediction)
                st.subheader("**Food Summary:**")
                st.write(summary)
            except Exception as e:
                st.error(f"Error in deep learning classifier: {str(e)}")

if __name__ == "__main__":
    main()
