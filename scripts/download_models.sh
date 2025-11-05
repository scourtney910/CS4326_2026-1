#!/bin/bash

# Download all required AI models

set -e

MODELS_DIR="./shared/models"

echo "======================================"
echo "Downloading AI Models"
echo "======================================"
echo ""

# Create models directory
mkdir -p "$MODELS_DIR"

# Download YOLOv8 model
echo "[1/3] Downloading YOLOv8 nano model..."
if [ ! -f "$MODELS_DIR/yolov8n.pt" ]; then
    wget -O "$MODELS_DIR/yolov8n.pt" \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    echo "YOLOv8n downloaded successfully"
else
    echo "YOLOv8n already exists"
fi

# Download Whisper model (will be downloaded by the model itself on first run)
echo "[2/3] Whisper models will be downloaded automatically on first run"
echo "    You can pre-download by running:"
echo "    python3 -c 'import whisper; whisper.load_model(\"base\")'"

# Download Llama model
echo "[3/3] Downloading Llama 3.2 1B model (quantized)..."
if [ ! -f "$MODELS_DIR/llama-3.2-1b-q4.gguf" ]; then
    echo "Please download Llama 3.2 1B quantized model manually:"
    echo "1. Visit https://huggingface.co/"
    echo "2. Search for 'llama-3.2-1b-instruct-q4_0.gguf' or similar quantized model"
    echo "3. Download and place in $MODELS_DIR/"
    echo ""
    echo "Alternative: Use a smaller model like Phi-3 mini"
    echo "Or download from: https://huggingface.co/TheBloke"
    echo ""
    echo "For testing, you can also use smaller/simpler models"
else
    echo "Llama model already exists"
fi

echo ""
echo "======================================"
echo "Model download complete!"
echo "======================================"
echo ""
echo "Note: Some models will be downloaded automatically on first run"
echo "Next step: Run 'make optimize-models' to optimize for Jetson"
echo ""
