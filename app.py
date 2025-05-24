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


# Create the Flask app
app = Flask(__name__)

# Path to the attendance CSV file
attendance_file = "attendance.csv"

# Initialize CSV file with headers if it doesn't exist
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
    
    # Read the image from the request
    file = request.files['image']
    img_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find face locations and encodings in the uploaded image
    face_locations = face_recognition.face_locations(rgb_image)
    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    recognized = []
    
    # Compare face encodings with the known encodings
    for enc in encodings:
        matches = face_recognition.compare_faces(known_encodings, enc)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            recognized.append(name)

            # Mark attendance for recognized person
            mark_attendance(name)
    
    return jsonify({"recognized": recognized})

def mark_attendance(name):
    """Log the recognized person's name and timestamp into the CSV file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp])

if __name__ == '__main__':
    app.run(debug=True)
