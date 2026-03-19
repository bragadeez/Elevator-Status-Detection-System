# Elevator Status Detection System

## Overview

This project presents a computer vision-based Elevator Status Detection System that detects elevator status and human presence using deep learning models. The system processes both images and videos to identify whether an elevator is open or closed and whether people are present inside.

The solution combines object detection and classification techniques to deliver real-time visual insights, making it applicable for smart building monitoring and safety systems.

---

## Features

- Detects elevator door state (open / closed)
- Identifies presence of people inside the elevator
- Supports both image and video input
- Generates annotated output with bounding boxes and labels
- Web interface for easy interaction and visualization

---

## Tech Stack

- Python
- Flask (Web Application Framework)
- OpenCV (Image and Video Processing)
- Ultralytics YOLO (Object Detection)
- TensorFlow / Deep Learning Models

---

## Project Structure

elevator-monitoring-system/
│
├── app/
│ ├── app.py
│ ├── static/
│ │ ├── styles.css
│ │ └── uploads/
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│
├── model/ # Model weights (not included in repo)
├── requirements.txt
├── README.md
├── .gitignore

---

## Model Details

- Model: YOLO-based detection system
- Dataset Size: ~3,600 images
- Classes:
  - Open Elevator
  - Closed Elevator
  - People in Elevator
  - No People in Elevator
- Performance: ~78% mAP@0.5

---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/bragadeez/elevator-monitoring-system.git

cd elevator-monitoring-system

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Request Model Weights

The trained model file is not included due to size constraints.
Mail me at bragadeeshwaranc@gmail.com to acquire the model weights

---

## Running the Application

cd app
python app.py

---

## Usage

1. Upload an image or video file
2. The system processes the input using the trained model
3. Output is displayed with detected objects and annotations

---

## Results

- Achieved ~78% mAP@0.5 on validation dataset
- Successfully detects elevator state and occupancy
- Works on both static images and video inputs

---

## Limitations

- Model performance may vary under poor lighting conditions
- Video processing can be slower due to frame-by-frame inference
- Requires manual download of model weights

---

## Future Improvements

- Deploy as a scalable API using FastAPI
- Integrate real-time streaming support
- Optimize inference speed using batching or GPU acceleration
- Add cloud deployment (AWS / GCP)
- Improve accuracy with larger and more diverse dataset

---

## Author

**Bragadeeshwaran C**  
Machine Learning Engineer | Computer Vision | GenAI

- GitHub: https://github.com/bragadeez
- LinkedIn: https://www.linkedin.com/in/bragadeeshwaran-c-107a05275/
