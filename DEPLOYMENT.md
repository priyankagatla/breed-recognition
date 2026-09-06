# Deployment Guide — Get a Permanent Public Link

Two free options, both give you a permanent URL anyone can open. **Option A
(Hugging Face Spaces)** is recommended because its free tier gives 16 GB RAM,
which comfortably runs TensorFlow. Option B (Render) is simpler but its free
tier only has 512 MB RAM, which is tight for TensorFlow — fine if you're
okay running in Demo Mode (no trained model) for the public link.

---

## Option A (Recommended): Hugging Face Spaces — Docker

Handles a real trained TensorFlow model without issues, 100% free, no
credit card required.

1. **Create a free account** at https://huggingface.co/join

2. **Create a new Space**:
   - Go to https://huggingface.co/new-space
   - Space name: e.g. `breed-recognition` (this becomes part of your URL)
   - License: any (e.g. MIT)
   - **Space SDK: choose "Docker"** → template "Blank"
   - Visibility: Public
   - Click **Create Space**

3. **Push this project into the Space's git repo.** The Space gives you a
   git URL like `https://huggingface.co/spaces/<your-username>/breed-recognition`.
   From inside the `breed_recognition/` project folder:
   ```bash
   git init
   git lfs install
   git lfs track "*.h5"          # only needed if you have a trained model file
   git add .
   git commit -m "Initial deploy"
   git remote add space https://huggingface.co/spaces/<your-username>/breed-recognition
   git push space main
   ```
   (You'll be asked to log in — use a Hugging Face **access token** as the
   password: create one at https://huggingface.co/settings/tokens with
   "write" access.)

4. Hugging Face automatically detects the `Dockerfile` in this project and
   builds + runs it. Build takes a few minutes the first time (installing
   TensorFlow). Watch progress under the **"Logs"** tab of your Space page.

5. Once it says **Running**, your permanent public URL is:
   ```
   https://<your-username>-breed-recognition.hf.space
   ```
   This is what you put in your report / share with evaluators. It stays up
   permanently and is free (the Space may go to sleep after ~48h of no
   traffic and take ~30s to wake up on the next visit — completely normal
   for free hosting, not a bug).

6. **To include your trained model**: after running `train_model.py`
   locally, make sure `model/breed_model.h5` and `model/class_indices.json`
   exist before you `git add . && git commit && git push space main` again.
   The app auto-detects them on startup and switches out of Demo Mode.

   > `breed_model.h5` is a binary file that can be several MB–tens of MB.
   > If it's over ~10 MB, use `git lfs track "*.h5"` (shown above) before
   > adding it, so git handles it properly.

---

## Option B: Render.com (simpler, best for Demo Mode / no TensorFlow)

1. Push this project to a **GitHub** repository (public or private).
2. Go to https://render.com → sign up free → **New +** → **Web Service**.
3. Connect your GitHub repo.
4. Render will detect `render.yaml` in this project automatically and
   pre-fill the build/start commands. If it doesn't, set manually:
   - **Build Command:** `pip install -r requirements.txt gunicorn`
   - **Start Command:** `gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Plan:** Free
5. Click **Create Web Service**. First deploy takes a few minutes.
6. Your permanent URL will look like:
   ```
   https://breed-recognition.onrender.com
   ```
   (exact name depends on what you called the service).

⚠️ If you've trained a real model and included `model/breed_model.h5` in
the repo, TensorFlow + the model may exceed Render's free 512 MB RAM limit,
causing the app to crash on startup ("Out of memory" in the logs). If that
happens, either:
   - Use Option A (Hugging Face Spaces) instead, or
   - Deploy without the `model/` folder so it runs in Demo Mode publicly
     (still shows the full working UI/UX — useful if your report already
     includes training results/screenshots separately).

---

## Which should I actually use for my submission?

- **Have a trained model and want live real predictions online?** → Option A
  (Hugging Face Spaces).
- **Just need the UI/UX reachable by a link, training results shown
  separately in your report?** → Either option works; Render is marginally
  quicker to set up if you're already comfortable with GitHub + Render.
