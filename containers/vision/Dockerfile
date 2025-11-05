# Vision Container - YOLO Person Detection
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopencv-dev \
    python3-opencv \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install \
    ultralytics \
    opencv-python \
    redis \
    numpy \
    pyyaml \
    python-dotenv

# Install TensorRT Python bindings (should be pre-installed in base image)
RUN pip3 install --upgrade tensorrt

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/

# Install additional requirements
RUN pip3 install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOCAPTURE_GSTREAMER=1

# Create non-root user for security
RUN useradd -m -u 1000 vision && chown -R vision:vision /app
USER vision

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run the application
CMD ["python3", "src/main.py"]
