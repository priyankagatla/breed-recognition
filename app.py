"""
app.py
------
Flask backend for "Image Based Breed Recognition for Cattle and Buffaloes of India".

Serves:
    GET  /                -> the web UI (templates/index.html)
    POST /predict          -> accepts an uploaded image, returns predicted breed
                              + confidence score + breed information (JSON)
    GET  /example/<name>   -> serves generated example thumbnail images
    GET  /health           -> simple health check

Model loading behaviour:
    - If model/breed_model.h5 and model/class_indices.json exist (produced by
      train_model.py), real predictions are made with TensorFlow/Keras.
    - If they don't exist yet, the app runs in DEMO MODE: it returns a
      deterministic, image-hash-based "prediction" from breed_info.py so the
      full UI/UX can be demoed and integrated end-to-end before training is done.
      A small banner in the API response (`"demo_mode": true`) marks this.
"""

import hashlib
import io
import json
import os
import random
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from breed_info import BREED_INFO, CLASS_NAMES, get_breed_info

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
UPLOAD_DIR = os.path.join(APP_DIR, "static", "uploads")
MODEL_PATH = os.path.join(MODEL_DIR, "breed_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

# --------------------------------------------------------------------------
# Try to load the real trained model. Fall back to demo mode if unavailable.
# --------------------------------------------------------------------------
MODEL = None
IDX_TO_CLASS = None
DEMO_MODE = True

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_INDICES_PATH):
        import numpy as np  # noqa: F401
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        MODEL = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_INDICES_PATH) as f:
            class_indices = json.load(f)
        IDX_TO_CLASS = {v: k for k, v in class_indices.items()}
        DEMO_MODE = False
        print(f"[app.py] Loaded trained model with {len(IDX_TO_CLASS)} classes.")
    else:
        print("[app.py] No trained model found in /model - starting in DEMO MODE.")
        print("[app.py] Train a model with train_model.py to enable real predictions.")
except Exception as e:  # pragma: no cover - defensive: never crash the server over model load issues
    print(f"[app.py] Could not load trained model ({e}). Falling back to DEMO MODE.")
    MODEL = None
    DEMO_MODE = True


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_real(image_bytes: bytes):
    """Run the actual trained CNN/Transfer-Learning model on the uploaded image."""
    import numpy as np
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = MODEL.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100
    breed_name = IDX_TO_CLASS.get(top_idx, "Unknown")
    return breed_name, confidence


def predict_demo(image_bytes: bytes):
    """
    DEMO MODE fallback: no trained model is present yet.
    Produces a *deterministic* (same image -> same result) pseudo-prediction
    so the UI/UX and integration can be fully tested and demoed end-to-end.
    Replace this by training a real model with train_model.py.
    """
    digest = hashlib.sha256(image_bytes).hexdigest()
    seed = int(digest[:8], 16)
    rng = random.Random(seed)

    breed_name = rng.choice(CLASS_NAMES)
    confidence = round(rng.uniform(85.0, 97.5), 1)
    return breed_name, confidence


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "demo_mode": DEMO_MODE,
                     "num_classes": len(IDX_TO_CLASS) if IDX_TO_CLASS else len(CLASS_NAMES)})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file part in the request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload JPG, JPEG or PNG."}), 400

    image_bytes = file.read()

    # Validate it's actually a readable image
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    # Save a copy so the frontend can display exactly what was uploaded
    ext = file.filename.rsplit(".", 1)[1].lower()
    saved_name = f"{uuid.uuid4().hex}.{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    with open(saved_path, "wb") as f:
        f.write(image_bytes)

    if MODEL is not None:
        breed_name, confidence = predict_real(image_bytes)
    else:
        breed_name, confidence = predict_demo(image_bytes)

    info = get_breed_info(breed_name)

    if confidence >= 85:
        confidence_label = "High Confidence"
        confidence_msg = "The model is highly confident about this prediction."
    elif confidence >= 60:
        confidence_label = "Moderate Confidence"
        confidence_msg = "The model is fairly confident, but consider a clearer image for best results."
    else:
        confidence_label = "Low Confidence"
        confidence_msg = "The model is not very confident. Try a clearer, well-lit image of the animal."

    return jsonify({
        "success": True,
        "demo_mode": DEMO_MODE,
        "breed": breed_name,
        "confidence": round(confidence, 1),
        "confidence_label": confidence_label,
        "confidence_message": confidence_msg,
        "species": info["species"],
        "origin": info["origin"],
        "uses": info["uses"],
        "characteristics": info["characteristics"],
        "description": info["description"],
        "image_url": f"/static/uploads/{saved_name}",
    })


@app.route("/breeds")
def breeds():
    """Returns the full breed database - useful for a 'Browse Breeds' page."""
    return jsonify(BREED_INFO)


if __name__ == "__main__":
    # PORT is read from the environment so this works unchanged on cloud hosts
    # (Hugging Face Spaces uses 7860, Render/Railway inject their own PORT).
    # Locally, with no PORT set, it still defaults to 5000 as before.
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
