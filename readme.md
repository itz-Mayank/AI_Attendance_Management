# AI Attendance Management

A web-based face attendance system that uses the webcam to capture a real-time image, recognizes faces using pre-saved encodings, and logs attendance with timestamps into a CSV file.

## 📦 Features

- Real-time webcam capture via browser
- Face recognition using `face_recognition` library
- Attendance logging to `attendance.csv`
- Simple Flask-based web interface

## 📁 Project Structure

```
Face-Attendance-System/
├── app.py                    # Flask application
├── face_encodings.pkl       # Pre-generated face encodings
├── attendance.csv           # Attendance log file
├── Students_images/         # Folder containing student face images
├── templates/
│   └── index.html           # HTML interface
├── generate_encodings.py    # Script to generate face_encodings.pkl
└── README.md                # Project documentation
```

## 🧠 How It Works

1. **Training Phase**:
   - Face images (named as `name_#.jpg`) are stored in `Students_images/`.
   - `face_encodings.pkl` is generated using a script that extracts encodings and saves them.

2. **Recognition Phase**:
   - User captures an image from webcam on the web interface.
   - Image is sent to the Flask server.
   - Server detects faces, matches with known encodings.
   - If matched, the name and timestamp are logged in `attendance.csv`.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/itz-Mayank/AI_Attendance_Management.git
cd AI_Attendance_Management
```

### 2. Install Dependencies

Make sure you have Python 3.7+ and pip installed.

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install manually:

```bash
pip install flask face_recognition opencv-python joblib
```

### 3. Prepare Face Encodings

Place your labeled images in the `Students_images/` folder. Then run:

```bash
python generate_encodings.py
```

This will generate `face_encodings.pkl`.

### 4. Run the Flask App

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## ✅ Attendance Output

Attendance will be saved in `attendance.csv` like so:

```
Name,Timestamp
john,2025-04-20 10:05:32
mary,2025-04-20 10:06:45
```

## 📸 Frontend Preview

The interface includes:

- Live webcam stream
- "Capture & Recognize" button
- Result display with recognized names

## 🛠 Technologies Used

- Python
- Flask
- OpenCV
- face_recognition
- HTML5 / JavaScript

## 🧩 Future Improvements

- Add daily duplicate check for attendance
- Admin dashboard to view attendance logs
- Notification system for late or absent entries
- Face registration via web interface

## 🤝 Contributing

Feel free to fork this repo, submit issues or PRs. All contributions are welcome!

## 📄 License

MIT License
