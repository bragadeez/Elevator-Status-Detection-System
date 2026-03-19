import os
import cv2
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

model = YOLO('model/best_2.pt')

# Helper function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to process the uploaded video
def process_video(video_path):
    output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')

    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the width, height, and FPS of the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}")  # Debug print
    
    # Check if FPS is valid
    if fps == 0 or frame_width == 0 or frame_height == 0:
        print("Error: Invalid frame properties. FPS or dimensions might be 0.")
        return None

    # Define codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0  # Counter to track how many frames are processed

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video. Total frames processed: {frame_count}")
            break

        # Get predictions using YOLO
        results = model(frame)
        annotated_frame = results[0].plot()  # This returns the frame with bounding boxes and labels

        # Write the annotated frame to the output video
        out.write(annotated_frame)
        frame_count += 1
    
    # Properly release video resources
    cap.release()
    out.release()

    print(f"Video saved successfully at: {output_video_path}")
    return output_video_path

# Function to process images
def process_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    annotated_img = results[0].plot()  # Add bounding boxes and labels
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
    cv2.imwrite(output_image_path, annotated_img)
    return output_image_path

# Homepage to upload files
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.endswith('.mp4'):  # Video file
            output_path = process_video(file_path)
            if output_path:
                return render_template('result.html', media_type='video', output_file='uploads/output.mp4')
            else:
                return 'Error processing video file.'
        else:  # Image file
            output_image_path = process_image(file_path)
            return render_template('result.html', media_type='image', output_file='uploads/output_image.jpg')
    
    return 'File type not allowed.'

# Route to serve the static files (for displaying output)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
