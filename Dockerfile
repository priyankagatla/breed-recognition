# Dockerfile for deploying the Breed Recognition app to Hugging Face Spaces
# (Docker SDK) or any other container host (Render, Railway, Fly.io, etc.)
#
# Hugging Face Spaces expects the app to listen on port 7860 - that's set
# via the PORT env var below, which app.py already reads.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by Pillow / TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Make sure runtime folders exist even on a fresh clone
RUN mkdir -p static/uploads model

ENV PORT=7860
ENV FLASK_DEBUG=false
EXPOSE 7860

# 2 workers is plenty for a small demo app; --timeout 120 gives slow CPU
# inference (TensorFlow on a free CPU instance) room to finish.
CMD ["sh", "-c", "gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:${PORT}"]
