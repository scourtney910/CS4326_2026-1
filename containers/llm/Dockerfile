# LLM Container - Sentiment Analysis with Llama
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip

# Install llama-cpp-python with CUDA support
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN pip3 install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Install other dependencies
RUN pip3 install \
    redis \
    numpy \
    pyyaml \
    python-dotenv \
    transformers \
    sentencepiece

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/

# Install additional requirements
RUN pip3 install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create non-root user for security
RUN useradd -m -u 1000 llm && chown -R llm:llm /app
USER llm

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run the application
CMD ["python3", "src/main.py"]
