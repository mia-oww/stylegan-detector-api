from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from skimage.feature import graycomatrix

app = Flask(__name__)


png_model = tf.keras.models.load_model('models/png_model/ImageDetectmodel.h5')
jpg_model = tf.keras.models.load_model('models/jpg_model/ImageDetectmodel.h5')

def getCoMatrices(img):
    b,g,r = cv2.split(img)
    distance = 1
    angle = 0
    rcomatrix = graycomatrix(r, [distance], [angle])
    gcomatrix = graycomatrix(g, [distance], [angle])
    bcomatrix = graycomatrix(b, [distance], [angle])
    tensor = tf.constant([rcomatrix[:,:,0,0], gcomatrix[:,:,0,0], bcomatrix[:,:,0,0]])
    tensor = tf.reshape(tensor, [256, 256, 3])
    return tensor

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    filename = file.filename.lower()

    # Read image file as numpy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Resize to 1024x1024 (your training size)
    img = cv2.resize(img, (1024, 1024))
    
    # Prepare input
    comat = getCoMatrices(img)
    comat = tf.reshape(comat, [1, 256, 256, 3])

    if filename.endswith('.png'):
        pred = np.argmax(png_model.predict(comat), axis=-1)[0]
    elif filename.endswith(('.jpg', '.jpeg')):
        pred = np.argmax(jpg_model.predict(comat), axis=-1)[0]
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    label = "Fake" if pred == 0 else "Real"
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
