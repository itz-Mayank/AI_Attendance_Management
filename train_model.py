# Import required libraries
import os
import cv2
import face_recognition
import numpy as np
import joblib

# Define dataset directory
dataset_dir = "Dataset"
known_encodings = []
known_names = []

print("Starting encoding process...")

# Iterate through each image in the dataset
for image_name in os.listdir(dataset_dir):
    if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(dataset_dir, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not read image: {image_name}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations and encode
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) == 1:
            name = image_name.split("_")[0].lower()
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"Encoded: {image_name} as {name}")
        else:
            print(f"Skipping {image_name}: Found {len(encodings)} faces")

# Save encoded faces
if known_encodings:
    data = {"encodings": known_encodings, "names": known_names}
    file_path = 'C:/Users/Mayank Meghwal/Desktop/face_encodings.pkl'
    
    joblib.dump(data, file_path)
    print(f"Encodings saved successfully at: {file_path}")
else:
    print("No valid face encodings found. Please check your dataset.")
