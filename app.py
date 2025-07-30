from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO('best.pt')

# Inisialisasi webcam (ganti 0 kalau pakai kamera eksternal berbeda)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Deteksi pakai YOLO
        results = model.predict(frame, conf=0.5, verbose=False)
        result = results[0]
        rendered = result.plot()

        # Encode ke JPEG
        ret, buffer = cv2.imencode('.jpg', rendered)
        frame = buffer.tobytes()

        # Stream frame (multipart)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('cam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5050)
