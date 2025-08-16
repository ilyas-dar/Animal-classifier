# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image
import os

# Step 1: Process images to extract histogram features
animals = ['cat', 'dog', 'horse', 'cow', 'sheep', 'pig', 'goat', 'chicken', 'duck', 'rabbit']
def process_image(image_path):
    """Extract histogram features from an image."""
    try:
        img = Image.open(image_path).resize((64, 64)).convert('RGB')
        img_array = np.array(img)
        # Compute histograms for R, G, B channels (16 bins each)
        hist_r, _ = np.histogram(img_array[:, :, 0], bins=16, range=(0, 255))
        hist_g, _ = np.histogram(img_array[:, :, 1], bins=16, range=(0, 255))
        hist_b, _ = np.histogram(img_array[:, :, 2], bins=16, range=(0, 255))
        # Concatenate histograms (48 features total)
        return np.concatenate([hist_r, hist_g, hist_b]).astype(float)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Load images from animal_dataset folder
data = []
labels = []
dataset_path = 'animal_dataset'  # Relative to L:\ilyas\myFirstModel\
for animal in animals:
    animal_folder = os.path.join(dataset_path, animal)
    if not os.path.exists(animal_folder):
        print(f"Folder {animal_folder} not found. Please create it and add images.")
        exit()
    for img_name in os.listdir(animal_folder):
        img_path = os.path.join(animal_folder, img_name)
        features = process_image(img_path)
        if features is not None:
            data.append(features)
            labels.append(animal)

# Check if data was loaded
if not data:
    print("No valid images found. Please check your animal_dataset folder.")
    exit()

# Create DataFrame
feature_columns = [f'hist_{i}' for i in range(48)]  # 16 bins x 3 channels
df = pd.DataFrame(data, columns=feature_columns)
df['animal'] = labels
df.to_csv('animal_data.csv', index=False)
print("Created dataset: animal_data.csv")

# Step 2: Load and preprocess data
X = df[feature_columns]
y = df['animal']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")

# Step 4: Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Step 5: Sample prediction
# Process a test image (replace with your own image path)
test_image_path = 'animal_dataset/cat/test_cat.jpg'  # Example
if os.path.exists(test_image_path):
    sample_features = process_image(test_image_path)
    if sample_features is not None:
        sample_scaled = scaler.transform([sample_features])
        prediction = model.predict(sample_scaled)
        print(f"Predicted animal for {test_image_path}: {prediction[0]}")
else:
    print("Test image not found. Skipping sample prediction.")

# Step 6: Save the model and scaler
joblib.dump(model, 'animal_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved as animal_classifier.pkl and scaler.pkl")