# Jetson Multi-Modal Edge AI Framework

A real-time multi-modal AI system running on NVIDIA Jetson Orin Nano that combines computer vision, audio processing, and natural language understanding for intelligent edge inference.

## System Overview

This framework detects speakers in video streams, transcribes their speech, analyzes sentiment for hostility, and triggers GPIO actions based on the analysis - all running locally on edge hardware.

## Architecture

### Core Pipeline
```
Camera → Person Detection (YOLO) → Active Speaker Detection
   ↓                                        ↓
Audio Stream → Speech Recognition → Sentiment Analysis → GPIO Trigger
```

### Components

1. **Vision Module** (`containers/vision/`)
   - YOLOv8 person detection and tracking
   - Bounding box coordinates extraction
   - Frame timestamping and synchronization

2. **Audio Module** (`containers/audio/`)
   - Audio capture and preprocessing
   - Speech-to-text (Whisper model)
   - Voice Activity Detection (VAD)

3. **Active Speaker Detection** (`containers/speaker_detection/`)
   - Audio-visual synchronization
   - Lip movement correlation
   - Speaker identification and tracking

4. **LLM Module** (`containers/llm/`)
   - Local inference (Llama 3.2 / Phi-3)
   - Sentiment and hostility classification
   - Quantized model support (4-bit/8-bit)

5. **Orchestrator** (`containers/orchestrator/`)
   - Event coordination
   - Inter-module communication (Redis/ZMQ)
   - GPIO control interface
   - Timestamping and logging

## Docker Architecture

Each module runs in an isolated container with GPU access:
- **Base Image**: `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`
- **GPU Acceleration**: CUDA, TensorRT, cuDNN
- **Communication**: Redis for pub/sub, shared volumes for media streams
- **Orchestration**: Docker Compose with GPU runtime

## Hardware Requirements

- NVIDIA Jetson Orin Nano (8GB recommended)
- USB Camera or CSI camera module
- USB Microphone or audio interface
- GPIO-compatible relay module
- JetPack 5.1+ installed

## Key Features

- **Real-time Processing**: <500ms end-to-end latency target
- **Local Inference**: All models run on-device, no cloud dependency
- **Modular Design**: Easy to swap models and components
- **GPU Optimized**: TensorRT acceleration for all models
- **Event-Driven**: Asynchronous processing pipeline
- **Production Ready**: Logging, monitoring, and error handling

## Technology Stack

- **Computer Vision**: YOLOv8 (TensorRT optimized)
- **Speech Recognition**: OpenAI Whisper (ONNX/TensorRT)
- **Active Speaker Detection**: Audio-visual correlation analysis
- **LLM**: Llama 3.2 1B/3B or Phi-3 Mini (quantized)
- **Orchestration**: Python 3.10, Redis, ZMQ
- **GPIO**: Jetson.GPIO library
- **Containers**: Docker with NVIDIA runtime

## Project Structure

```
jetson-multimodal-framework/
├── containers/
│   ├── vision/              # YOLO person detection
│   ├── audio/               # Speech-to-text processing
│   ├── speaker_detection/   # Active speaker identification
│   ├── llm/                 # Sentiment analysis
│   └── orchestrator/        # System coordination & GPIO
├── shared/
│   ├── models/              # Optimized model weights
│   ├── data/                # Shared data volumes
│   └── config/              # Configuration files
├── scripts/
│   ├── setup_jetson.sh      # Jetson environment setup
│   ├── download_models.sh   # Model download script
│   └── optimize_models.py   # TensorRT conversion
├── tests/
│   ├── integration/         # End-to-end tests
│   └── unit/                # Module tests
├── docker-compose.yml       # Multi-container orchestration
├── Makefile                 # Build and deployment commands
└── README.md
```

## Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd jetson-multimodal-framework

# 2. Setup Jetson environment
make setup-jetson

# 3. Download and optimize models
make download-models
make optimize-models

# 4. Build Docker containers
make build

# 5. Run the system
make run

# 6. Monitor logs
make logs
```

## Performance Targets

| Component | Latency | FPS/Throughput |
|-----------|---------|----------------|
| YOLO Detection | <50ms | 20 FPS |
| Speech Recognition | <200ms | Real-time |
| Speaker Detection | <100ms | 10 FPS |
| LLM Inference | <200ms | ~5 tokens/sec |
| **End-to-End** | **<500ms** | **Real-time** |

## Development Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
- [x] Project structure and documentation
- [ ] Docker base images setup
- [ ] Inter-container communication (Redis)
- [ ] Basic orchestrator framework

### Phase 2: Vision Pipeline (Week 2-3)
- [ ] YOLOv8 integration and TensorRT optimization
- [ ] Camera capture module
- [ ] Person tracking implementation

### Phase 3: Audio Pipeline (Week 3-4)
- [ ] Whisper model integration
- [ ] Audio capture and VAD
- [ ] Active speaker detection algorithm

### Phase 4: LLM Integration (Week 4-5)
- [ ] LLM model selection and quantization
- [ ] Sentiment analysis pipeline
- [ ] Hostility classification logic

### Phase 5: Integration & Testing (Week 5-6)
- [ ] End-to-end pipeline integration
- [ ] GPIO relay control
- [ ] Performance optimization
- [ ] System testing and validation

## Configuration

Key configuration files:
- `shared/config/vision.yaml` - YOLO settings
- `shared/config/audio.yaml` - Whisper and audio settings
- `shared/config/llm.yaml` - LLM inference settings
- `shared/config/orchestrator.yaml` - System coordination
- `shared/config/gpio.yaml` - GPIO pin mappings

## Monitoring and Debugging

```bash
# View all container logs
make logs

# Monitor system resources
make monitor

# Debug specific module
make debug MODULE=vision

# Performance profiling
make profile
```

## Safety and Security

- Input validation on all data pipelines
- Rate limiting on GPIO triggers
- Fail-safe mechanisms for hardware control
- Audit logging for all hostility detections
- Configurable sensitivity thresholds

## Contributing

See `CONTRIBUTING.md` for development guidelines.

## License

[To be determined]

## Authors

CS4326_2026-1 Team

## Acknowledgments

- NVIDIA Jetson community
- OpenAI Whisper team
- Ultralytics YOLOv8
- Active speaker detection research community
