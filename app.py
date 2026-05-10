"""
Cats vs Dogs — Web Demo
Flask API + HTML frontend for image classification
Usage: python app.py  →  open http://127.0.0.1:5000
"""
import os, io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf

app = Flask(__name__, static_folder=".")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "output", "model.keras")
IMG_SIZE = (160, 160)

print("Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model ready!")


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Preprocess
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    # Predict
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "dog" if prob >= 0.5 else "cat"
    conf = prob if prob >= 0.5 else 1 - prob

    return jsonify({
        "label": label,
        "confidence": round(conf * 100, 2),
        "cat_prob": round((1 - prob) * 100, 2),
        "dog_prob": round(prob * 100, 2),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
