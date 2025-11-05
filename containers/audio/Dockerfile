# Audio Container - Speech-to-Text with Whisper
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install \
    openai-whisper \
    pyaudio \
    redis \
    numpy \
    pyyaml \
    python-dotenv \
    webrtcvad \
    scipy \
    librosa

# Install ONNX Runtime for GPU acceleration
RUN pip3 install onnxruntime-gpu

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/

# Install additional requirements
RUN pip3 install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd -m -u 1000 audio && chown -R audio:audio /app
USER audio

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run the application
CMD ["python3", "src/main.py"]
