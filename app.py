from flask import Flask, Response, request, jsonify
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from flask_cors import CORS  # Import the CORS module

import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)
FRAME_INTERVAL = 5

model = YOLO("weights/best.pt")

# def process_video(video_file):
#    model.predict(video_file, save=True, imgsz=320, conf=0.5, stream=True)

# this function returns base64 encoded image with bboxes
def process_image(image):
    results = model.predict(image)
    for result in results:
        annotator = Annotator(image)
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
                
    img = annotator.result()  
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')
def stream2cv(stream):
    image_stream = np.fromstring(stream, np.uint8)
    return cv2.imdecode(image_stream, cv2.IMREAD_COLOR)

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = stream2cv(file.stream.read())
        encoded_img_str = process_image(image)
        response = jsonify({'image': encoded_img_str, 'message': 'Image processed successfully'})
        # That's only because of the CORS
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    except Exception as e:
        return jsonify({'error': str(e)})
@app.route('/upload_video', methods=['POST'])
def upload_video():
    return 410
if __name__ == '__main__':
    app.run(debug=True)
