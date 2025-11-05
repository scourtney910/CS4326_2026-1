# Getting Started Guide

This guide will help you set up and run the Jetson Multi-Modal Edge AI Framework on your NVIDIA Jetson Orin Nano.

## Prerequisites

### Hardware Requirements

- **NVIDIA Jetson Orin Nano** (8GB recommended)
- **JetPack 5.1+** installed
- **USB Camera** or CSI camera module
- **USB Microphone** or audio interface
- **GPIO-compatible relay module** (for hardware control)
- **MicroSD card** (64GB+ recommended)
- **Power supply** (appropriate for your Jetson model)

### Software Requirements

- JetPack SDK (includes CUDA, cuDNN, TensorRT)
- Docker and Docker Compose
- Python 3.8+
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd jetson-multimodal-framework
```

### 2. Setup Jetson Environment

Run the setup script to install all dependencies:

```bash
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

This script will:
- Update system packages
- Install Docker and Docker Compose
- Configure NVIDIA Container Runtime
- Install system dependencies
- Create directory structure

**Important:** After setup, log out and back in for Docker group changes to take effect.

### 3. Download AI Models

Download required models:

```bash
make download-models
```

This downloads:
- YOLOv8 nano model for person detection
- Prepares for Whisper model (auto-downloaded on first run)
- Instructions for Llama 3.2 model

**Note:** You'll need to manually download the Llama 3.2 1B quantized model (GGUF format) from Hugging Face. See the [Model Download Guide](#model-download-guide) below.

### 4. Optimize Models for Jetson

Convert models to optimized formats (TensorRT, ONNX):

```bash
make optimize-models
```

This may take 10-15 minutes, especially for TensorRT conversion.

### 5. Configure the System

Edit configuration files in `shared/config/` to match your hardware:

**shared/config/orchestrator.yaml:**
```yaml
gpio:
  relay_pin: 7  # Change to your GPIO pin number
```

**shared/config/vision.yaml:**
```yaml
camera:
  device_index: 0  # Change if using different camera
  type: usb       # or 'csi' for CSI camera
```

**shared/config/audio.yaml:**
```yaml
audio:
  device_index: 0  # Change if using different microphone
```

### 6. Build Docker Containers

Build all containers:

```bash
make build
```

This will take 15-30 minutes on first build.

### 7. Test Hardware

Before running the full system, test your hardware:

**Test camera:**
```bash
v4l2-ctl --list-devices
# Note your camera device (usually /dev/video0)
```

**Test microphone:**
```bash
arecord -l
# Note your audio device index
```

**Test GPIO (optional):**
```bash
sudo python3 -c "import Jetson.GPIO as GPIO; print('GPIO available')"
```

### 8. Run the System

Start all services:

```bash
make run
```

This starts all containers in the background.

### 9. Monitor Logs

View logs from all containers:

```bash
make logs
```

Or monitor specific containers:

```bash
make logs-vision
make logs-audio
make logs-llm
make logs-orchestrator
```

### 10. Test the System

1. **Stand in front of the camera** - You should see detection logs
2. **Speak into the microphone** - Watch for transcription logs
3. **Say something neutral** - "Hello, how are you?" (should classify as SAFE)
4. **Check GPIO behavior** - If configured, test with hostile phrase (carefully!)

## Model Download Guide

### Downloading Llama 3.2 1B (Quantized)

You have several options:

#### Option 1: Hugging Face (Recommended)

1. Visit [Hugging Face](https://huggingface.co/)
2. Search for "llama-3.2-1b-instruct-q4_0 gguf"
3. Download a 4-bit quantized GGUF model (look for "TheBloke" models)
4. Place in `shared/models/` directory
5. Rename to `llama-3.2-1b-q4.gguf`

Example:
```bash
cd shared/models
wget <huggingface-download-url>
mv <downloaded-file>.gguf llama-3.2-1b-q4.gguf
```

#### Option 2: Alternative Models

If you can't find Llama 3.2, use alternatives:

- **Phi-3 Mini** (3.8B, quantized) - Great performance
- **Llama 2 7B** (quantized) - Larger but good quality
- **TinyLlama 1.1B** - Smaller, faster

Download GGUF format models optimized for llama.cpp.

#### Option 3: Quantize Your Own

If you have a non-quantized model:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
make

# Convert and quantize
python3 convert.py /path/to/model --outtype q4_0
```

## Verification

Check that everything is running:

```bash
make status
```

Expected output:
```
Container Status:
jetson-redis          running
jetson-vision         running
jetson-audio          running
jetson-speaker-detection running
jetson-llm            running
jetson-orchestrator   running

Redis Status:
PONG

GPU Status:
[nvidia-smi output]
```

## Troubleshooting

### Container won't start

Check logs:
```bash
docker-compose logs <container-name>
```

Common issues:
- **GPU not accessible**: Ensure NVIDIA runtime is configured
- **Camera not found**: Check device index and permissions
- **Audio not found**: Check ALSA configuration
- **Model not found**: Verify model paths in config files

### Low FPS / High Latency

1. **Check GPU usage:**
   ```bash
   tegrastats
   ```

2. **Reduce model sizes:**
   - Use `yolov8n` instead of larger YOLO models
   - Use Whisper `tiny` or `base` models
   - Use smaller LLM (1B instead of 3B)

3. **Adjust frame rate:**
   - Lower camera FPS in `shared/config/vision.yaml`
   - Process every N frames: `skip_frames: 2`

### GPIO Not Working

1. **Check permissions:**
   ```bash
   sudo usermod -aG gpio $USER
   ```

2. **Verify pin number:**
   - Use BOARD numbering (physical pin numbers)
   - Check Jetson pinout diagram

3. **Test manually:**
   ```bash
   sudo python3 -c "
   import Jetson.GPIO as GPIO
   GPIO.setmode(GPIO.BOARD)
   GPIO.setup(7, GPIO.OUT)
   GPIO.output(7, GPIO.HIGH)
   "
   ```

### False Positives / Negatives

Adjust thresholds in `shared/config/llm.yaml`:

```yaml
classification:
  min_confidence: 0.8  # Increase to reduce false positives
```

### Audio Quality Issues

1. **Test microphone:**
   ```bash
   arecord -d 5 test.wav
   aplay test.wav
   ```

2. **Adjust VAD sensitivity:**
   Edit `shared/config/audio.yaml`:
   ```yaml
   vad:
     aggressiveness: 3  # More aggressive filtering
   ```

## Common Commands

```bash
# Start system
make run

# Stop system
make stop

# Restart system
make restart

# View all logs
make logs

# Monitor resources
make monitor

# Shell into container
make debug MODULE=vision

# Run tests
make test

# Clean everything
make clean
```

## Next Steps

1. **Fine-tune Configuration** - Adjust thresholds for your use case
2. **Test Thoroughly** - Test in various lighting and noise conditions
3. **Add Monitoring** - Set up monitoring dashboard
4. **Improve Models** - Upgrade to better models as needed
5. **Implement Features** - Add custom features (see [ARCHITECTURE.md](ARCHITECTURE.md))

## Safety Warnings

⚠️ **Important Safety Considerations:**

1. **Test thoroughly** before deploying in production
2. **Set appropriate thresholds** to avoid false positives
3. **Implement emergency stop** mechanisms
4. **Log all actions** for audit trail
5. **Comply with privacy laws** regarding audio/video recording
6. **Secure the system** - this processes sensitive data
7. **Test GPIO control** carefully to avoid hardware damage

## Support

For issues, questions, or contributions:

1. Check existing documentation in `docs/`
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Check container logs for errors
4. File an issue in the project repository

## Performance Benchmarks

Expected performance on Jetson Orin Nano 8GB:

| Component | Target | Typical |
|-----------|--------|---------|
| Vision (YOLO) | 20 FPS | 15-20 FPS |
| Audio (Whisper) | Real-time | <200ms latency |
| Speaker Detection | 10 FPS | 8-10 FPS |
| LLM Inference | <200ms | 150-250ms |
| **End-to-End** | **<500ms** | **300-400ms** |

Memory usage:
- Vision: ~2GB GPU
- Audio: ~1GB GPU
- LLM: ~3GB GPU
- Total: ~6GB GPU + 2GB RAM

## Additional Resources

- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-orin-nano)
- [TensorRT Optimization Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [Whisper Documentation](https://github.com/openai/whisper)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)

---

**Ready to build?** Start with Step 1 above and work through each step carefully. Good luck! 🚀
