"""
Trains a deep learning model for food classification using EfficientNet-B0.
This script handles model training, evaluation, and prediction.
Training was done on Google Colab and code was update to scripts and moved to local machine.
Directory structure was updated to be integreated with current code.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from huggingface_hub import hf_hub_download

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.append(project_root)

# Constants for model configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names list
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

# Directory paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data/final/splits")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Add this after the imports
model = None

def setup_data(data_dir):
    """Setup data loaders and transformations for training, validation, and testing.
    This function:
    1. Defines data augmentation transforms for training
    2. Sets up validation/test transforms
    3. Creates datasets and data loaders
    Args:
        data_dir (str): Root directory containing train/val/test splits
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    # Training data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train/images"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val/images"), transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test/images"), transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names

def create_model(num_classes):
    """Create and initialize the EfficientNet-B0 model.
    Args:
        num_classes (int): Number of food classes
    Returns:
        torch.nn.Module: Initialized model
    """
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)
    return model

def train_model(model, train_loader, val_loader, model_path, history_path):
    """Train the model and save the best version based on validation accuracy.
    
    This function: Sets up loss function and optimizer, 
    Trains and validates for specified number of epochs,
    and saves best model and training history
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        model_path (str): Path to save best model
        history_path (str): Path to save training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss /= total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # calculate loss and accuracy
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        # print epoch results
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")

    # Save training history
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")

def evaluate_model(model, test_loader, class_names):
    """Evaluate the model on test data.
    This function:
    1. Calculates accuracy
    2. Generates classification report
    3. Creates and saves confusion matrix
    
    Args:
        model (torch.nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        class_names (list): List of class names
        
    Returns:
        tuple: (classification report, confusion matrix, accuracy)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix
    script_dir = os.path.dirname(os.path.abspath(__file__))
    confusion_matrix_path = os.path.join(script_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Confusion matrix saved to {confusion_matrix_path}")
    
    return report, cm, accuracy

def load_model():
    """Load the model from Hugging Face once when the app starts."""
    global model
    if model is None:
        print("Loading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="okamdar/food-classification",
            filename="efficientnet_b0.pth",
            local_dir=MODELS_DIR
        )
        
        # Create and load model
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        model = model.to(DEVICE)
    return model

def predict_from_image(image_array):
    """Predict food class from an image array using the loaded model.
    
    Args:
        image_array: numpy array of the image (RGB format)
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    # Get the loaded model
    model = load_model()
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return class_names[predicted_idx.item()], confidence.item()

def main():
    """Main function to orchestrate the training and evaluation process."""
    print(f"Using device: {DEVICE}")
    
    # Setup data
    train_loader, val_loader, test_loader, class_names = setup_data(DATA_DIR)
    print(f"Classes: {len(class_names)}")
    
    # Create and train model
    print("\nCreating model...")
    model = create_model(len(class_names))
    print("Model created")
    
    # Train the model
    print("\nStarting training...")
    train_model(model, train_loader, val_loader, 
                os.path.join(MODELS_DIR, "efficientnet_b0.pth"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_history.json"))
    
    # Evaluate on test set
    print("\nStarting evaluation...")
    evaluate_model(model, test_loader, class_names)
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main() 