import os
import cv2
import face_recognition
import numpy as np
import joblib

dataset_dir = "Students_images"
known_encodings = []
known_names = []

# Encode each image
for image_name in os.listdir(dataset_dir):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image_path = os.path.join(dataset_dir, image_name)
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) == 1:
            name = image_name.split("_")[0].lower()
            known_encodings.append(encodings[0])
            known_names.append(name)
        else:
            print(f"Skipping {image_name}: Found {len(encodings)} faces")

# Save the data
data = {"encodings": known_encodings, "names": known_names}
joblib.dump(data, "face_encodings.pkl")
print("✅ Face encodings saved successfully.")
