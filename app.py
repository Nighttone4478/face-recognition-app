import cv2
import numpy as np
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.getcwd(), 'trained_model.yml')
CASCADE_PATH = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')

# 加載預訓練的 LBPH 模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)  # 訓練好的模型

# 檢查是否存在分類器文件
if not os.path.exists(CASCADE_PATH):
    raise Exception("Haar cascade file not found!")
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image part"}), 400

    # 讀取圖像數據
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 假設使用灰度圖像進行人臉識別
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 偵測人臉並進行識別
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"status": "fail", "message": "No face detected"})

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)
        if confidence < 100:
            return jsonify({"status": "success", "message": "Face recognized", "label": label})
    
    return jsonify({"status": "fail", "message": "Face not recognized"})

if __name__ == '__main__':
    # 使用 Heroku 的動態端口
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)