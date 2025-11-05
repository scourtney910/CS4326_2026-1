"""
Audio Container - Speech Recognition with Whisper

This module captures audio from microphone, detects voice activity,
transcribes speech using Whisper, and publishes transcription events to Redis.
"""

import os
import time
import json
import redis
import numpy as np
import whisper
import pyaudio
import webrtcvad
from typing import Optional
import logging
from collections import deque
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio capture and speech-to-text transcription"""

    def __init__(self):
        self.redis_client = None
        self.model = None
        self.audio_interface = None
        self.stream = None
        self.vad = None

        # Load configuration from environment
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.model_path = os.getenv('MODEL_PATH', 'base')
        self.audio_device_index = int(os.getenv('AUDIO_DEVICE_INDEX', 0))
        self.language = os.getenv('LANGUAGE', 'en')
        self.vad_threshold = float(os.getenv('VAD_THRESHOLD', 0.5))

        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration_ms = 30  # VAD works with 10, 20, or 30 ms
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        self.buffer_duration_s = 1.0  # Buffer 1 second before/after speech

        # Speech detection
        self.speech_buffer = deque(maxlen=int(self.buffer_duration_s * 1000 / self.chunk_duration_ms))
        self.is_speaking = False
        self.silence_chunks = 0
        self.silence_threshold = 20  # Number of silent chunks before stopping

        logger.info("Audio Processor initialized")

    def connect_redis(self):
        """Connect to Redis message broker"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=False  # Keep as bytes for audio data
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_path}")
            self.model = whisper.load_model(self.model_path)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def initialize_audio(self):
        """Initialize audio capture and VAD"""
        try:
            # Initialize PyAudio
            self.audio_interface = pyaudio.PyAudio()

            # List available devices (for debugging)
            logger.info("Available audio devices:")
            for i in range(self.audio_interface.get_device_count()):
                info = self.audio_interface.get_device_info_by_index(i)
                logger.info(f"  Device {i}: {info['name']}")

            # Open audio stream
            self.stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.audio_device_index,
                frames_per_buffer=self.chunk_size
            )

            # Initialize VAD
            self.vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3, 2 is moderate

            logger.info("Audio capture initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            raise

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Check if audio chunk contains speech using VAD

        Args:
            audio_chunk: Raw audio bytes

        Returns:
            True if speech detected, False otherwise
        """
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[dict]:
        """
        Transcribe audio using Whisper

        Args:
            audio_data: Audio samples as numpy array

        Returns:
            Transcription result dict or None
        """
        try:
            # Whisper expects float32 normalized to [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Transcribe
            result = self.model.transcribe(
                audio_float,
                language=self.language,
                fp16=False  # Set to True if GPU supports FP16
            )

            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def publish_transcription(self, transcription: str, audio_chunk_id: str,
                            start_time: float, end_time: float, confidence: float):
        """Publish transcription event to Redis"""
        event = {
            'timestamp': time.time(),
            'component': 'audio',
            'event_type': 'transcription',
            'data': {
                'audio_chunk_id': audio_chunk_id,
                'transcription': transcription,
                'confidence': confidence,
                'start_time': start_time,
                'end_time': end_time,
                'language': self.language
            }
        }

        try:
            self.redis_client.xadd(
                'stream:transcriptions',
                {'payload': json.dumps(event)}
            )
            logger.info(f"Transcription published: '{transcription}'")
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")

    def run(self):
        """Main processing loop"""
        logger.info("Starting audio processing loop...")
        logger.info("Listening for speech...")

        audio_accumulator = []
        speech_start_time = None

        try:
            while True:
                # Read audio chunk
                audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)

                # Check for speech
                has_speech = self.is_speech(audio_chunk)

                # Convert to numpy array
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                self.speech_buffer.append(audio_array)

                if has_speech:
                    if not self.is_speaking:
                        # Speech started
                        self.is_speaking = True
                        speech_start_time = time.time()
                        logger.info("Speech detected, recording...")

                        # Add buffered audio before speech
                        audio_accumulator = list(self.speech_buffer)
                    else:
                        # Continue recording
                        audio_accumulator.append(audio_array)

                    self.silence_chunks = 0

                elif self.is_speaking:
                    # Potential silence during speech
                    self.silence_chunks += 1
                    audio_accumulator.append(audio_array)

                    if self.silence_chunks >= self.silence_threshold:
                        # Speech ended
                        self.is_speaking = False
                        speech_end_time = time.time()

                        logger.info("Speech ended, transcribing...")

                        # Concatenate audio chunks
                        audio_data = np.concatenate(audio_accumulator)

                        # Transcribe
                        result = self.transcribe_audio(audio_data)

                        if result and result['text'].strip():
                            audio_chunk_id = str(uuid.uuid4())

                            # Publish transcription
                            self.publish_transcription(
                                transcription=result['text'].strip(),
                                audio_chunk_id=audio_chunk_id,
                                start_time=speech_start_time,
                                end_time=speech_end_time,
                                confidence=1.0  # Whisper doesn't provide confidence
                            )

                        # Reset
                        audio_accumulator = []
                        self.silence_chunks = 0
                        logger.info("Ready for next speech...")

        except KeyboardInterrupt:
            logger.info("Shutting down audio processor...")
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio_interface is not None:
            self.audio_interface.terminate()
        logger.info("Audio processor shut down cleanly")


def main():
    """Main entry point"""
    processor = AudioProcessor()

    # Initialize components
    processor.connect_redis()
    processor.load_model()
    processor.initialize_audio()

    # Run processing loop
    processor.run()


if __name__ == '__main__':
    main()
