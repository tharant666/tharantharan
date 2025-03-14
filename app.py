from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the model file exists
model_path = "best (8).pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Check the path!")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('index.html', uploaded_image=filename)
    return redirect(request.url)

@app.route('/process', methods=['POST'])
def process_image():
    filename = request.form['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(filepath)

    if image is None:
        return "Error: Could not read the image.", 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    zoom_factor = 2.0
    height, width = image_rgb.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    zoomed_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    results = model(zoomed_image)
    good_paddy_count = 0
    green_paddy_count = 0
    CONFIDENCE_THRESHOLD = 0.35

    for *xyxy, conf, cls in results.xyxy[0]:
        conf = conf.item()
        class_index = int(cls)
        class_name = model.names[class_index].strip()

        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, xyxy)

        if class_name == "G":
            good_paddy_count += 1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_region_size = 10
            center_region = zoomed_image[
                center_y - center_region_size // 2 : center_y + center_region_size // 2,
                center_x - center_region_size // 2 : center_x + center_region_size // 2,
            ]
            hsv_center = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
            lower_green_hsv = np.array([25, 40, 40])
            upper_green_hsv = np.array([95, 255, 255])
            mask_hsv = cv2.inRange(hsv_center, lower_green_hsv, upper_green_hsv)
            green_pixels = cv2.countNonZero(mask_hsv)
            total_pixels = center_region.shape[0] * center_region.shape[1]

            if green_pixels / total_pixels > 0.5:
                green_paddy_count += 1
                cv2.rectangle(zoomed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(zoomed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    total_paddy_count = good_paddy_count
    green_paddy_percentage = (green_paddy_count / total_paddy_count) * 100 if total_paddy_count > 0 else 0

    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
    cv2.imwrite(output_image_path, cv2.cvtColor(zoomed_image, cv2.COLOR_RGB2BGR))

    return render_template('index.html', uploaded_image=filename, processed_image='processed_' + filename,
                          total_paddy=total_paddy_count, green_paddy=green_paddy_count,
                          green_paddy_percentage=green_paddy_percentage)

if __name__ == '__main__':
    app.run(debug=True)