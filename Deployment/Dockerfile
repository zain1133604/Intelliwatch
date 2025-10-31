# ==============================================
# 🚀 SMART CCTV — Dockerfile
# ==============================================

# ---- Base Image ----
# (Use NVIDIA’s PyTorch image for GPU support; change to "python:3.11-slim" if CPU only)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# ---- System Setup ----
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level deps (for OpenCV, ffmpeg, etc.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Working Directory ----
WORKDIR /app

# ---- Copy Requirements ----
COPY requirements.txt .

# ---- Install Python Dependencies ----
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Copy App Code ----
COPY . .

# ---- Environment Variables ----
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
RUN mkdir -p /tmp/hf_cache

# ---- Expose Port ----
EXPOSE 7860

# ---- Command to Run the App ----
# (You can change --host 0.0.0.0 --port 7860 to whatever you like)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
