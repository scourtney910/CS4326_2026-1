# System Architecture Document

## Overview

This document describes the technical architecture of the Jetson Multi-Modal Edge AI Framework, designed for real-time person detection, active speaker identification, speech transcription, sentiment analysis, and automated response triggering.

## System Design Principles

1. **Modularity**: Each component is independently deployable and testable
2. **Local-First**: All inference happens on-device, no cloud dependencies
3. **Real-Time**: Sub-500ms end-to-end latency for critical path
4. **Fault Tolerant**: Graceful degradation when components fail
5. **GPU Optimized**: Maximum utilization of Jetson hardware acceleration

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Jetson Orin Nano Device                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │   Camera     │    │  Microphone  │    │   GPIO Relay    │   │
│  │   Input      │    │    Input     │    │    Output       │   │
│  └──────┬───────┘    └──────┬───────┘    └────────▲────────┘   │
│         │                   │                      │            │
│         │                   │                      │            │
│  ┌──────▼────────────────────▼──────────────────────┴────────┐  │
│  │                  Docker Network Bridge                     │  │
│  │                    (redis-network)                         │  │
│  └──┬──────────┬──────────┬──────────┬──────────┬───────────┘  │
│     │          │          │          │          │               │
│  ┌──▼──────┐ ┌▼────────┐ ┌▼────────┐ ┌▼───────┐ ┌▼──────────┐  │
│  │ Vision  │ │  Audio  │ │ Speaker │ │  LLM   │ │Orchestrator│ │
│  │Container│ │Container│ │Detection│ │Container│ │ Container  │ │
│  │         │ │         │ │Container│ │        │ │            │ │
│  │ YOLOv8  │ │ Whisper │ │ AV Sync │ │Llama3.2│ │ Coordinator│ │
│  │TensorRT │ │  ONNX   │ │ Models  │ │  4bit  │ │   +GPIO    │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ └───┬────┘ └─────┬──────┘  │
│       │           │           │          │            │          │
│  ┌────▼───────────▼───────────▼──────────▼────────────▼──────┐  │
│  │                   Redis Message Broker                      │  │
│  │         (Pub/Sub + Streams for Event Coordination)          │  │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Shared Docker Volumes                          │  │
│  │  /models  /data/video  /data/audio  /config  /logs         │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Vision Container (Person Detection & Tracking)

**Purpose**: Detect and track persons in video stream, provide bounding boxes

**Technology Stack**:
- YOLOv8n/YOLOv8s (optimized for Jetson)
- TensorRT for GPU acceleration
- OpenCV for video capture
- ByteTrack or DeepSORT for tracking

**Inputs**:
- Video stream from camera (CSI/USB)
- Configuration: model path, confidence threshold, IOU threshold

**Outputs** (to Redis):
```json
{
  "timestamp": 1234567890.123,
  "frame_id": 12345,
  "detections": [
    {
      "track_id": 1,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "center": [cx, cy]
    }
  ]
}
```

**Performance Target**: 20 FPS @ 640x640 resolution

### 2. Audio Container (Speech Recognition)

**Purpose**: Capture audio, detect voice activity, transcribe speech

**Technology Stack**:
- OpenAI Whisper (tiny/base model, ONNX optimized)
- PyAudio/PortAudio for audio capture
- WebRTC VAD or Silero VAD
- Audio preprocessing (noise reduction)

**Inputs**:
- Audio stream from microphone
- Configuration: model size, language, VAD sensitivity

**Outputs** (to Redis):
```json
{
  "timestamp": 1234567890.123,
  "audio_chunk_id": "abc123",
  "transcription": "the text that was spoken",
  "confidence": 0.87,
  "start_time": 1234567888.5,
  "end_time": 1234567890.5,
  "language": "en"
}
```

**Performance Target**: Real-time transcription with <200ms delay

### 3. Speaker Detection Container (Active Speaker Identification)

**Purpose**: Correlate audio activity with video to identify who is speaking

**Technology Stack**:
- Audio-visual synchronization algorithms
- Lip movement detection (optional: lightweight lip detection model)
- Temporal correlation analysis
- Audio energy mapping to video regions

**Approach Options**:

**Option A: Audio-Visual Correlation (Lightweight)**
- Compute audio energy/activity per time window
- Correlate with motion in mouth region of detected persons
- Use temporal consistency for robust detection

**Option B: Deep Learning Model (More Accurate)**
- Lightweight active speaker detection model (e.g., TalkNet, Light-ASD)
- Crop face regions from YOLO detections
- Feed audio+video to model for classification

**Inputs**:
- Person detections from Vision container
- Audio activity from Audio container
- Synchronized video frames

**Outputs** (to Redis):
```json
{
  "timestamp": 1234567890.123,
  "active_speaker": {
    "track_id": 1,
    "confidence": 0.89,
    "bbox": [x1, y1, x2, y2],
    "audio_chunk_id": "abc123"
  }
}
```

**Performance Target**: 10 FPS speaker detection

### 4. LLM Container (Sentiment & Hostility Analysis)

**Purpose**: Analyze transcribed text for hostile content

**Technology Stack**:
- Llama 3.2 1B/3B (quantized 4-bit) OR Phi-3 Mini
- llama.cpp or vLLM for optimized inference
- Prompt engineering for binary classification
- Batch processing for efficiency

**Prompt Strategy**:
```
System: You are a safety classifier. Classify the following speech as HOSTILE or SAFE.
HOSTILE includes: threats, aggressive language, verbal abuse, hate speech.
SAFE includes: normal conversation, questions, neutral statements.

User: [transcribed text]
Assistant: Classification:
```

**Inputs**:
- Transcription events from Audio container
- Configuration: model path, temperature, classification threshold

**Outputs** (to Redis):
```json
{
  "timestamp": 1234567890.123,
  "audio_chunk_id": "abc123",
  "transcription": "the text that was analyzed",
  "classification": "HOSTILE",
  "confidence": 0.92,
  "reasoning": "Contains threatening language"
}
```

**Performance Target**: <200ms inference per text chunk

### 5. Orchestrator Container (System Coordination & GPIO Control)

**Purpose**: Coordinate all components, maintain state, trigger GPIO actions

**Technology Stack**:
- Python asyncio for event-driven coordination
- Redis client for pub/sub and streams
- Jetson.GPIO for hardware control
- State machine for tracking pipeline flow
- SQLite for event logging

**Responsibilities**:

1. **Event Coordination**
   - Subscribe to all component outputs
   - Match transcriptions with speaker detections
   - Correlate LLM results with person bounding boxes
   - Maintain temporal consistency

2. **State Management**
   - Track active speakers over time
   - Buffer recent detections for correlation
   - Handle out-of-order events
   - Implement timeout mechanisms

3. **GPIO Control**
   - Trigger relay when hostile speaker detected
   - Debouncing and rate limiting
   - Safety timeouts (auto-off after N seconds)
   - Manual override capability

4. **Logging & Monitoring**
   - Event logging to SQLite
   - Performance metrics collection
   - Error handling and recovery
   - Health checks for all containers

**Data Flow State Machine**:
```
State 1: DETECTING_PERSONS
  → Person detected → State 2

State 2: LISTENING_FOR_SPEECH
  → Speech detected → State 3
  → Timeout → State 1

State 3: IDENTIFYING_SPEAKER
  → Speaker identified → State 4
  → No match → State 1

State 4: ANALYZING_SENTIMENT
  → Analysis complete → State 5

State 5: DECISION
  → If HOSTILE → Trigger GPIO → State 6
  → If SAFE → State 1

State 6: GPIO_ACTIVE
  → Timeout → State 1
  → Manual override → State 1
```

**Configuration**:
```yaml
gpio:
  relay_pin: 7  # GPIO pin number
  active_high: true
  trigger_duration: 5.0  # seconds
  cooldown_period: 2.0  # seconds between triggers

correlation:
  max_time_diff: 1.0  # seconds between audio and speaker detection
  buffer_size: 100  # events to buffer for correlation
  min_confidence: 0.7  # minimum confidence for action

safety:
  require_confirmation: false
  max_triggers_per_minute: 10
  enable_audit_log: true
```

## Data Flow Example

**Scenario**: Person speaks hostile phrase

```
T=0.000s: Vision detects person
  → Redis: detection_events stream
  {track_id: 1, bbox: [100, 50, 300, 400], timestamp: 0.000}

T=0.050s: Audio captures speech
  → Redis: audio_events stream
  {audio_chunk_id: "abc123", raw_audio: [...], timestamp: 0.050}

T=0.100s: Speaker detection correlates
  → Redis: speaker_events stream
  {track_id: 1, audio_chunk_id: "abc123", confidence: 0.91, timestamp: 0.100}

T=0.200s: Audio container transcribes
  → Redis: transcription_events stream
  {audio_chunk_id: "abc123", text: "I will harm you", timestamp: 0.200}

T=0.350s: LLM analyzes sentiment
  → Redis: llm_events stream
  {audio_chunk_id: "abc123", classification: "HOSTILE", confidence: 0.95, timestamp: 0.350}

T=0.360s: Orchestrator correlates all events
  → Match track_id 1 with audio_chunk_id "abc123"
  → Classification is HOSTILE
  → Trigger GPIO relay
  → Log event: {track_id: 1, bbox: [100, 50, 300, 400], text: "...", action: "GPIO_TRIGGERED"}

T=5.360s: GPIO auto-off after 5 second duration
```

**Total Latency**: ~360ms (well within 500ms target)

## Communication Protocol

### Redis Streams

Each component publishes to dedicated streams:

- `stream:detections` - Person detections from Vision
- `stream:audio` - Raw audio chunks and VAD results
- `stream:transcriptions` - Speech-to-text results
- `stream:speakers` - Active speaker identifications
- `stream:sentiment` - LLM classification results
- `stream:actions` - GPIO trigger events

### Message Format

All messages follow this structure:
```json
{
  "timestamp": 1234567890.123,  // Unix timestamp with milliseconds
  "component": "vision",         // Source component
  "event_type": "detection",     // Event type
  "sequence_id": 12345,          // Auto-incrementing sequence
  "data": { ... }                // Component-specific payload
}
```

### Synchronization Strategy

**Time Sync**: All containers use system clock (NTP synced)
**Buffering**: Orchestrator buffers 1 second of events for correlation
**Matching**: Events correlated using timestamp proximity and IDs

## Docker Container Details

### Base Image Strategy

All containers derive from:
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```

This provides:
- CUDA 11.4
- cuDNN 8.6
- TensorRT 8.5
- PyTorch 2.0
- Python 3.10

### GPU Access

Docker Compose configuration:
```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=all
```

### Resource Allocation

Jetson Orin Nano (8GB version):
- Vision Container: 2GB GPU memory, 2 CPU cores
- Audio Container: 1GB GPU memory, 1 CPU core
- Speaker Detection: 1GB GPU memory, 1 CPU core
- LLM Container: 3GB GPU memory, 2 CPU cores
- Orchestrator: 1GB RAM, 2 CPU cores
- Redis: 512MB RAM, 1 CPU core

### Volume Mounts

```yaml
volumes:
  models:        # Shared model weights
  data_video:    # Video frame buffer
  data_audio:    # Audio chunk buffer
  config:        # Configuration files
  logs:          # Application logs
```

## Model Optimization

### YOLOv8 → TensorRT

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='engine', device=0, half=True)  # FP16
# Result: yolov8n.engine (2-3x faster than PyTorch)
```

### Whisper → ONNX

```python
import whisper
from whisper.onnx import export_onnx

model = whisper.load_model("base")
export_onnx(model, "whisper_base.onnx", quantize=True)
# Use with onnxruntime-gpu for acceleration
```

### Llama 3.2 → 4-bit Quantization

```bash
# Using llama.cpp for quantization
python convert.py models/llama-3.2-1b --outtype q4_0
# Result: 4-bit quantized model (~700MB for 1B model)
```

## Performance Optimization Strategies

### 1. Pipeline Parallelism
- All containers run simultaneously
- Non-blocking communication via Redis
- Async I/O throughout

### 2. Batching
- LLM processes multiple transcriptions in batch
- Vision processes frames in batches of 2-4
- Audio chunks processed in streaming fashion

### 3. Model Quantization
- FP16 for vision models
- INT8/INT4 for LLM
- Dynamic quantization for Whisper

### 4. Hardware Acceleration
- TensorRT for all CNNs
- CUDA for preprocessing
- GPU-accelerated audio processing

### 5. Memory Management
- Shared memory for video frames (avoid copies)
- Memory pooling for audio buffers
- Efficient tensor management

## Error Handling

### Container Failure Recovery

- **Health Checks**: Each container exposes /health endpoint
- **Restart Policy**: `restart: unless-stopped`
- **Graceful Degradation**: System continues with reduced functionality
- **Logging**: All errors logged to centralized location

### Edge Cases

1. **No person detected**: System stays in DETECTING_PERSONS state
2. **Speech without visible speaker**: Log event, no GPIO trigger
3. **LLM unavailable**: Queue transcriptions, process when available
4. **GPIO failure**: Log error, continue detection
5. **Camera/mic disconnection**: Attempt reconnection every 5s

## Security Considerations

### Data Privacy
- All processing on-device, no external communication
- Optional encryption of stored logs
- Configurable data retention policies

### Safety Mechanisms
- Rate limiting on GPIO triggers
- Confidence thresholds for classification
- Manual override capability
- Emergency stop feature

### Access Control
- Docker containers run as non-root users
- GPIO access via specific group permissions
- Configuration file permissions restricted

## Monitoring and Debugging

### Metrics Collection

Key metrics exposed via Redis:
- Frame processing rate (FPS)
- Audio transcription latency
- LLM inference time
- End-to-end pipeline latency
- GPU memory usage
- CPU usage per container

### Debugging Tools

```bash
# View real-time events
redis-cli XREAD STREAMS stream:detections 0

# Monitor container logs
docker-compose logs -f vision

# Check GPU usage
tegrastats

# Profile specific container
nsys profile docker exec vision python main.py
```

## Testing Strategy

### Unit Tests
- Each component has isolated tests
- Mock Redis for testing
- Test model loading and inference

### Integration Tests
- Test event flow between containers
- Verify correlation logic
- Test GPIO triggering (with mock hardware)

### Performance Tests
- Measure end-to-end latency
- Stress test with continuous input
- Memory leak detection

### Hardware-in-Loop Tests
- Test with real camera and microphone
- Verify GPIO relay operation
- Test in various lighting/noise conditions

## Future Enhancements

1. **Multi-person tracking**: Track multiple speakers simultaneously
2. **Speaker recognition**: Identify known individuals
3. **Gesture detection**: Incorporate body language analysis
4. **Cloud sync**: Optional cloud backup of events
5. **Mobile app**: Remote monitoring and control
6. **Improved models**: Upgrade to larger/better models as hardware allows

## References

- NVIDIA Jetson Documentation: https://developer.nvidia.com/embedded/jetson-orin-nano
- TensorRT Optimization: https://docs.nvidia.com/deeplearning/tensorrt/
- Whisper Model: https://github.com/openai/whisper
- YOLOv8: https://github.com/ultralytics/ultralytics
- Active Speaker Detection: Research papers on audio-visual synchronization