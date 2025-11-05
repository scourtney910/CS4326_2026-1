# Quick Reference Guide

## Essential Commands

### Setup & Build
```bash
# Initial setup (run once)
make setup-jetson
make download-models
make optimize-models
make build

# Start the system
make run

# Stop the system
make stop
```

### Monitoring
```bash
# View all logs
make logs

# View specific container logs
make logs-vision
make logs-audio
make logs-speaker
make logs-llm
make logs-orchestrator

# Monitor system resources
make monitor

# Check status
make status
```

### Debugging
```bash
# Shell into a container
make debug MODULE=vision

# Or use docker exec directly
docker exec -it jetson-vision /bin/bash

# View Redis streams
docker exec -it jetson-redis redis-cli
XREAD STREAMS stream:detections 0
```

### Development
```bash
# Rebuild after code changes
make build

# Restart specific container
docker-compose restart vision

# View container logs in real-time
docker-compose logs -f vision
```

## Configuration Files

### Vision Settings
```yaml
# shared/config/vision.yaml
camera:
  device_index: 0        # Change for different camera
  width: 640
  height: 640

model:
  confidence_threshold: 0.5  # Person detection confidence
```

### Audio Settings
```yaml
# shared/config/audio.yaml
audio:
  device_index: 0        # Change for different microphone
  sample_rate: 16000

model:
  model_size: base       # tiny, base, small, medium
```

### LLM Settings
```yaml
# shared/config/llm.yaml
model:
  path: /models/llama-3.2-1b-q4.gguf

inference:
  temperature: 0.1       # Lower = more deterministic
```

### GPIO Settings
```yaml
# shared/config/orchestrator.yaml
gpio:
  relay_pin: 7           # GPIO pin number (BOARD mode)
  trigger_duration: 5.0  # Seconds

correlation:
  min_confidence: 0.7    # Minimum confidence to trigger
```

## Common Issues & Fixes

### Camera Not Found
```bash
# List cameras
v4l2-ctl --list-devices

# Test camera
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg

# Update config
nano shared/config/vision.yaml
```

### Microphone Not Found
```bash
# List audio devices
arecord -l

# Test microphone
arecord -d 5 test.wav
aplay test.wav

# Update config
nano shared/config/audio.yaml
```

### GPU Not Accessible
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvcr.io/nvidia/l4t-base:r35.2.1 nvidia-smi

# Reinstall NVIDIA runtime
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### Container Fails to Start
```bash
# Check logs
docker-compose logs <container-name>

# Check Redis
docker exec jetson-redis redis-cli ping

# Restart all
make restart
```

### High Latency
```bash
# Check GPU usage
tegrastats

# Reduce camera FPS
# Edit shared/config/vision.yaml
fps: 15

# Use smaller models
# Vision: yolov8n (already smallest)
# Audio: whisper tiny instead of base
# LLM: Use 1B instead of 3B model
```

## Redis Streams Reference

### Available Streams
- `stream:detections` - Person detections from Vision
- `stream:transcriptions` - Speech-to-text from Audio
- `stream:speakers` - Active speaker IDs
- `stream:sentiment` - Sentiment analysis results
- `stream:actions` - GPIO trigger events

### View Stream Data
```bash
# Shell into Redis
docker exec -it jetson-redis redis-cli

# Read from stream
XREAD STREAMS stream:detections 0

# Get stream length
XLEN stream:detections

# Get last entry
XREVRANGE stream:detections + - COUNT 1
```

## GPIO Pin Reference (Jetson Orin Nano)

### BOARD Mode Pin Numbers
```
Pin 7  - GPIO 216
Pin 11 - GPIO 50
Pin 12 - GPIO 79
Pin 13 - GPIO 14
Pin 15 - GPIO 194
Pin 16 - GPIO 232
Pin 18 - GPIO 15
Pin 19 - GPIO 16
```

### Test GPIO
```bash
# Test pin 7
sudo python3 << EOF
import Jetson.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.output(7, GPIO.HIGH)
print("GPIO 7 HIGH")
input("Press Enter to turn OFF...")
GPIO.output(7, GPIO.LOW)
GPIO.cleanup()
EOF
```

## Performance Tuning

### Reduce GPU Memory Usage
```bash
# Edit docker-compose.yml
# Reduce model sizes in configs
```

### Increase FPS
```yaml
# shared/config/vision.yaml
performance:
  skip_frames: 1  # Process every frame
```

### Reduce Latency
- Use TensorRT models (run `make optimize-models`)
- Reduce camera resolution
- Use smaller LLM model
- Adjust batch sizes

## Helpful Docker Commands

```bash
# List all containers
docker-compose ps

# Stop all containers
docker-compose down

# Rebuild specific container
docker-compose build vision

# View container resource usage
docker stats

# Clean up everything
docker-compose down -v
docker system prune -a
```

## File Locations

```
/home/user/CS4326_2026-1/
├── shared/
│   ├── models/          # Put model files here
│   ├── data/            # Runtime data
│   ├── config/          # Edit configs here
│   └── logs/            # Check logs here
├── containers/          # Container source code
└── scripts/             # Utility scripts
```

## Model Files Checklist

- [ ] `shared/models/yolov8n.pt` - YOLO PyTorch model
- [ ] `shared/models/yolov8n.engine` - YOLO TensorRT engine (after optimization)
- [ ] Whisper model (auto-downloaded on first run)
- [ ] `shared/models/llama-3.2-1b-q4.gguf` - LLM model (manual download)

## Testing Workflow

1. **Test camera:**
   ```bash
   make shell-vision
   python3 << EOF
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   print(f"Camera working: {ret}")
   EOF
   ```

2. **Test microphone:**
   ```bash
   make shell-audio
   python3 << EOF
   import pyaudio
   p = pyaudio.PyAudio()
   print(f"Found {p.get_device_count()} audio devices")
   EOF
   ```

3. **Test Redis:**
   ```bash
   make shell-redis
   PING
   ```

4. **Test full pipeline:**
   - Start system: `make run`
   - Stand in front of camera
   - Speak: "Hello, this is a test"
   - Check orchestrator logs: `make logs-orchestrator`

## Emergency Stop

### Stop GPIO Immediately
```bash
# Shell into orchestrator
docker exec -it jetson-orchestrator python3 << EOF
import Jetson.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.output(7, GPIO.LOW)
GPIO.cleanup()
print("GPIO disabled")
EOF
```

### Disable Auto-Start
```yaml
# docker-compose.yml
# Change: restart: unless-stopped
# To:     restart: "no"
```

## Backup & Restore

### Backup Configuration
```bash
tar -czf config-backup.tar.gz shared/config/
```

### Backup Logs
```bash
tar -czf logs-backup.tar.gz shared/logs/
```

### Restore
```bash
tar -xzf config-backup.tar.gz
```

## Contact & Support

- **Documentation:** See README.md, ARCHITECTURE.md, GETTING_STARTED.md
- **Logs:** Check `shared/logs/` directory
- **Issues:** File an issue in the repository

---

**Quick Start:** `make setup-jetson && make download-models && make optimize-models && make build && make run`
