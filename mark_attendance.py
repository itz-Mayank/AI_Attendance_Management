import os
import cv2
import csv
import joblib
import face_recognition
import numpy as np

# Load known encodings
data = joblib.load("face_encodings.pkl")
known_encodings = data["encodings"]
known_names = data["names"]

def recognize_faces(image_path, tolerance=0.5):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    recognized = set()

    for encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        min_distance = min(distances)

        if min_distance < tolerance:
            match_index = np.argmin(distances)
            recognized.add(known_names[match_index])
        else:
            print("⚠️ Unknown face detected, skipping...")

    return list(recognized)

def mark_attendance_csv(class_name, recognized_students):
    filename = f"{class_name}_attendance.csv"
    exists = os.path.exists(filename)

    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["Student Name", "Status"])

        for student in recognized_students:
            writer.writerow([student, "Present"])

# Example usage
image_path = "class_image_1.jpg"
recognized_students = recognize_faces(image_path)

print(f"✅ Recognized Students: {recognized_students}")
if recognized_students:
    mark_attendance_csv("Class_1", recognized_students)
else:
    print("❌ No known faces found.")
