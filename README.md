# Image Based Breed Recognition for Cattle and Buffaloes of India

A full-stack web application that identifies Indian cattle and buffalo
breeds from an uploaded photo using Computer Vision + Transfer Learning
(MobileNetV2 CNN), and displays the predicted breed, a confidence score, and
detailed breed information (origin, uses, characteristics, description) —
matching the project's methodology: **Image Upload → Preprocessing →
Feature Extraction → Breed Classification → Confidence & Breed Information →
User Interface**.

## Project structure

```
breed_recognition/
├── app.py                    # Flask backend: serves UI + /predict API
├── train_model.py            # Transfer-learning training script (MobileNetV2)
├── prepare_dataset.py        # Splits a raw image collection into train/val/test
├── download_real_images.py   # Fetches real breed photos from Wikipedia (run locally, needs internet)
├── breed_info.py             # Breed metadata (origin, uses, characteristics, description)
├── requirements.txt
├── dataset/
│   └── README.md             # Where/how to get & organize the dataset
├── model/                    # Trained model is saved here after training
│   ├── breed_model.h5              (created by train_model.py)
│   ├── class_indices.json          (created by train_model.py)
│   ├── training_history.png        (created by train_model.py)
│   └── confusion_matrix.png        (created by train_model.py, optional)
├── static/
│   ├── css/style.css         # UI styling (matches the reference design)
│   ├── js/script.js          # Upload / drag-drop / predict / download / share logic
│   ├── img/examples/         # 6 example thumbnail images shown in the sidebar
│   └── uploads/              # Uploaded images are temporarily saved here
└── templates/
    └── index.html            # Single-page UI
```

## 1. Install dependencies

```bash
cd breed_recognition
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Note: `tensorflow`, `matplotlib`, and `scikit-learn` are only needed for
> **training** (`train_model.py`). If you just want to run the web app in
> demo mode first (see below), `pip install flask pillow` is enough.

## 2. Run the app immediately — Demo Mode (no training required)

You can launch and fully test the entire UI/UX right away, before you've
trained a model:

```bash
python app.py
```

Open **http://localhost:5000** in your browser. Since no trained model
exists yet, the backend automatically runs in **DEMO MODE**: it returns a
deterministic, realistic-looking prediction (same image → same result) drawn
from the breed database in `breed_info.py`, so you can build/demo/screenshot
the full app end-to-end. Every API response includes `"demo_mode": true` so
you (and your evaluators) always know when real inference is active.

## Want a permanent public link anyone can open?

See **`DEPLOYMENT.md`** for step-by-step instructions to deploy this project
for free on Hugging Face Spaces (recommended — handles TensorFlow
comfortably) or Render.com.

## 3. Replace the placeholder example thumbnails with real photos

The 6 "Example Images" shipped in `static/img/examples/` are simple generated
placeholders (no internet access was available while assembling this
project). To swap them for real photos, run this **on your own machine**
(needs internet):

```bash
pip install requests
python download_real_images.py
```

This fetches the real lead photo for Gir, Ongole, Murrah, Jaffarabadi,
Kankrej, and Nili-Ravi straight from their Wikipedia articles (Wikimedia
Commons images, CC-BY-SA / public domain) and overwrites the placeholders —
no other code changes needed, the filenames already match what
`static/js/script.js` expects.

Want photos for all 30 breeds in `breed_info.py` (e.g. to also seed a
starter `raw_dataset/` folder for training)?

```bash
python download_real_images.py --all --seed-dataset
```

⚠️ One Wikipedia photo per breed is only a *seed* — nowhere near enough to
train an accurate CNN. You still need to collect many more real images per
breed (see `dataset/README.md`) before running `train_model.py`.

## 4. Get a dataset & train the real model

1. Read **`dataset/README.md`** for exact folder layout and dataset sources
   (Kaggle "Indian Bovine Breeds", Roboflow Universe, or your own field
   photos of cattle/buffalo).
2. If your images are in one flat folder-per-breed collection, split them:
   ```bash
   python prepare_dataset.py --raw raw_dataset --out dataset --train 0.7 --val 0.15
   ```
3. Train (stage 1 — frozen MobileNetV2 backbone, fast):
   ```bash
   python train_model.py --epochs 25 --batch_size 32
   ```
4. (Optional, recommended) Fine-tune stage 2 — unfreezes the last ~30
   backbone layers for a few epochs at a lower learning rate to squeeze out
   extra accuracy:
   ```bash
   python train_model.py --epochs 10 --fine_tune
   ```
5. This produces `model/breed_model.h5` + `model/class_indices.json`.
   Restart `python app.py` — it will detect the trained model automatically
   and switch out of demo mode into real predictions. Check `/health` to
   confirm: `{"status": "ok", "demo_mode": false, "num_classes": N}`.

## 5. Integration notes / how the pieces fit together

- **Frontend → Backend**: `static/js/script.js` posts the uploaded file as
  `multipart/form-data` to `POST /predict`. The response JSON drives every
  part of the Prediction Result screen (breed name, confidence ring %,
  origin/uses/characteristics/description) — no page reload needed.
- **Backend → Model**: `app.py` loads `model/breed_model.h5` +
  `model/class_indices.json` once at startup. Preprocessing at inference
  time (resize to 224×224, `mobilenet_v2.preprocess_input`) exactly matches
  what `train_model.py` uses during training — keep these in sync if you
  change the backbone.
- **Model → Breed Info**: The model's predicted class name is looked up in
  `breed_info.py`'s `BREED_INFO` dict to attach human-readable metadata.
  Add new breeds there (and retrain with matching dataset folders) to expand
  breed coverage.
- **Example images**: `static/img/examples/*.jpg` are generated placeholder
  thumbnails (labelled Gir, Ongole, Murrah, Jaffarabadi, Kankrej, Nili-Ravi).
  Swap them for real photos from your dataset any time — same filenames, or
  update the `EXAMPLES` array at the top of `script.js`.
- **Download / Share**: handled entirely client-side in `script.js` — no
  extra backend route needed (downloads a `.txt` summary; share uses the
  Web Share API with a clipboard-copy fallback).

## 6. Deploying

For a real deployment, run behind a production WSGI server instead of Flask's
dev server, e.g.:

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

Put a reverse proxy (nginx) in front for TLS/static-file caching if exposing
this publicly.

## 7. Tech stack summary (maps to the project report)

| Report requirement                         | Implementation                                   |
|---------------------------------------------|---------------------------------------------------|
| Computer Vision / AI / CNN / Transfer Learning | MobileNetV2 (ImageNet weights) + custom classification head, `train_model.py` |
| Image preprocessing & augmentation          | `ImageDataGenerator` (resize, rotate, flip, zoom, brightness) |
| Feature extraction                          | MobileNetV2 convolutional backbone (frozen, then fine-tuned) |
| Breed classification + confidence score     | Softmax output layer; top-1 class + probability, `app.py` |
| Breed information display                   | `breed_info.py` metadata database |
| User Interface (upload image → results)     | `templates/index.html`, `static/css`, `static/js` |
