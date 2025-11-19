FROM python:3.10-slim

WORKDIR /app

# Install system deps for OpenCV, V4L2 and Tkinter
RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    v4l-utils \
    libv4l-dev \
    libopencv-videoio-dev \
    libjpeg-dev \
    libpng-dev \
    libgtk-3-dev \
    python3-tk \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TFLite runtime (ARM-friendly)
RUN pip install --no-cache-dir tflite-runtime==2.14.0

# Pin numpy version for TFLite
RUN pip install --no-cache-dir "numpy<2"

# Install other Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model
COPY model.tflite .
COPY app.py .

CMD ["python3", "app.py"]
