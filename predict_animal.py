# Import required libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
import os

# Define the list of animals (must match training)
animals = ['cat', 'dog', 'horse', 'cow', 'sheep', 'pig', 'goat', 'chicken', 'duck', 'rabbit']

def process_image(image_path):
    """Extract histogram features from an image (same as training)."""
    try:
        img = Image.open(image_path).resize((64, 64)).convert('RGB')
        img_array = np.array(img)
        # Compute histograms for R, G, B channels (16 bins each, 48 features total)
        hist_r, _ = np.histogram(img_array[:, :, 0], bins=16, range=(0, 255))
        hist_g, _ = np.histogram(img_array[:, :, 1], bins=16, range=(0, 255))
        hist_b, _ = np.histogram(img_array[:, :, 2], bins=16, range=(0, 255))
        return np.concatenate([hist_r, hist_g, hist_b]).astype(float)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Load the trained model and scaler
try:
    model = joblib.load('animal_classifier.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("Error: animal_classifier.pkl or scaler.pkl not found. Please run the training script first.")
    exit()

# Specify the path to your test image
test_image_path = 'test_image.jpg'  # Replace with your image path, e.g., 'L:\\ilyas\\myFirstModel\\test_image.jpg'

# Process and predict
if os.path.exists(test_image_path):
    features = process_image(test_image_path)
    if features is not None:
        # Scale features (same as training)
        features_scaled = scaler.transform([features])
        # Predict
        prediction = model.predict(features_scaled)[0]
        print(f"Predicted animal for {test_image_path}: {prediction}")
    else:
        print(f"Failed to process {test_image_path}.")
else:
    print(f"Test image {test_image_path} not found. Please provide a valid image path.")