import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
import csv
from datetime import datetime

# Load face encodings and names
data = joblib.load("face_encodings.pkl")
known_encodings = data["encodings"]
known_names = data["names"]

# Initialize Flask app
app = Flask(__name__)

# CSV for attendance
attendance_file = "attendance.csv"

# Create CSV file with headers if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Timestamp'])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_image)
    encodings = face_recognition.face_encodings(rgb_image, face_locations)

    recognized = []
    threshold = 0.45  # Lower = stricter matching

    for face_encoding in encodings:
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        matches = np.where(face_distances < threshold)[0]

        if len(matches) > 0:
            best_match_idx = matches[np.argmin(face_distances[matches])]
            name = known_names[best_match_idx]

            if name not in recognized:
                recognized.append(name)
                mark_attendance(name)
        else:
            print("⚠️ Unknown face detected — not in dataset.")

    return jsonify({"recognized": recognized})

def mark_attendance(name):
    """Mark attendance in CSV with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp])
        print(f"✅ Attendance marked for {name} at {timestamp}")
    except Exception as e:
        print(f"❌ Error marking attendance: {e}")

if __name__ == '__main__':
    app.run(debug=True)
