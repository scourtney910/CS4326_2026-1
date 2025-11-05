"""
Speaker Detection Container - Active Speaker Identification

This module correlates audio activity with person detections to identify
who is speaking in the video stream using audio-visual synchronization.
"""

import os
import time
import json
import redis
import numpy as np
from typing import Dict, List, Optional
from collections import deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpeakerDetector:
    """Correlates audio with video to identify active speaker"""

    def __init__(self):
        self.redis_client = None

        # Load configuration from environment
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.correlation_window = float(os.getenv('CORRELATION_WINDOW', 1.0))
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.7))

        # Buffers for temporal correlation
        self.detection_buffer = deque(maxlen=100)
        self.transcription_buffer = deque(maxlen=50)

        logger.info("Speaker Detector initialized")

    def connect_redis(self):
        """Connect to Redis message broker"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def find_closest_detection(self, transcription_time: float) -> Optional[Dict]:
        """
        Find person detection closest in time to transcription

        Args:
            transcription_time: Timestamp of transcription

        Returns:
            Closest detection dict or None
        """
        closest_detection = None
        min_time_diff = float('inf')

        for detection in self.detection_buffer:
            time_diff = abs(detection['timestamp'] - transcription_time)

            if time_diff < self.correlation_window and time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_detection = detection

        return closest_detection

    def correlate_speaker(self, transcription_event: Dict) -> Optional[Dict]:
        """
        Correlate transcription with person detection to identify speaker

        This is a simplified approach based on temporal proximity.
        For more sophisticated detection, implement:
        - Lip movement detection
        - Audio direction estimation
        - Multiple speaker tracking

        Args:
            transcription_event: Transcription event data

        Returns:
            Speaker detection event or None
        """
        transcription_time = transcription_event['timestamp']
        audio_chunk_id = transcription_event['data']['audio_chunk_id']

        # Find closest person detection
        detection = self.find_closest_detection(transcription_time)

        if detection is None:
            logger.warning("No person detection found for transcription")
            return None

        # Check if only one person is detected (simple case)
        detections_data = detection['data']['detections']

        if len(detections_data) == 0:
            return None

        if len(detections_data) == 1:
            # Only one person, assume they are speaking
            speaker = detections_data[0]
            confidence = 0.9  # High confidence when only one person
        else:
            # Multiple persons detected
            # TODO: Implement more sophisticated speaker identification
            # For now, use the person closest to center or largest bbox
            speaker = self.select_most_likely_speaker(detections_data)
            confidence = 0.6  # Lower confidence with multiple people

        if confidence < self.confidence_threshold:
            return None

        speaker_event = {
            'timestamp': time.time(),
            'component': 'speaker_detection',
            'event_type': 'speaker',
            'data': {
                'active_speaker': {
                    'track_id': speaker['track_id'],
                    'bbox': speaker['bbox'],
                    'center': speaker['center'],
                    'confidence': confidence,
                    'audio_chunk_id': audio_chunk_id
                }
            }
        }

        return speaker_event

    def select_most_likely_speaker(self, detections: List[Dict]) -> Dict:
        """
        Select most likely speaker from multiple detections

        Currently uses simple heuristic: person with largest bounding box
        TODO: Implement audio-visual correlation, lip movement detection

        Args:
            detections: List of person detections

        Returns:
            Most likely speaker detection
        """
        largest_speaker = None
        largest_area = 0

        for detection in detections:
            bbox = detection['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            if area > largest_area:
                largest_area = area
                largest_speaker = detection

        return largest_speaker if largest_speaker else detections[0]

    def publish_speaker_event(self, speaker_event: Dict):
        """Publish speaker identification event to Redis"""
        try:
            self.redis_client.xadd(
                'stream:speakers',
                {'payload': json.dumps(speaker_event)}
            )
            logger.info(f"Speaker identified: track_id={speaker_event['data']['active_speaker']['track_id']}")
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")

    def process_detection_stream(self):
        """Subscribe to detection events and buffer them"""
        last_id = '0'
        while True:
            try:
                streams = self.redis_client.xread(
                    {'stream:detections': last_id},
                    count=10,
                    block=100
                )

                if streams:
                    for stream_name, messages in streams:
                        for message_id, message_data in messages:
                            payload = json.loads(message_data['payload'])
                            self.detection_buffer.append(payload)
                            last_id = message_id

            except Exception as e:
                logger.error(f"Error reading detection stream: {e}")
                time.sleep(1)

    def process_transcription_stream(self):
        """Subscribe to transcription events and correlate with detections"""
        last_id = '0'
        while True:
            try:
                streams = self.redis_client.xread(
                    {'stream:transcriptions': last_id},
                    count=1,
                    block=100
                )

                if streams:
                    for stream_name, messages in streams:
                        for message_id, message_data in messages:
                            payload = json.loads(message_data['payload'])

                            # Correlate speaker
                            speaker_event = self.correlate_speaker(payload)

                            if speaker_event:
                                self.publish_speaker_event(speaker_event)

                            last_id = message_id

            except Exception as e:
                logger.error(f"Error reading transcription stream: {e}")
                time.sleep(1)

    def run(self):
        """Main processing loop"""
        logger.info("Starting speaker detection...")
        logger.info("Listening to detection and transcription streams...")

        # In a real implementation, use threading or asyncio for concurrent stream processing
        # For simplicity, we'll use a single-threaded approach alternating between streams

        detection_last_id = '0'
        transcription_last_id = '0'

        try:
            while True:
                # Read detections
                try:
                    detection_streams = self.redis_client.xread(
                        {'stream:detections': detection_last_id},
                        count=10,
                        block=50
                    )

                    if detection_streams:
                        for stream_name, messages in detection_streams:
                            for message_id, message_data in messages:
                                payload = json.loads(message_data['payload'])
                                self.detection_buffer.append(payload)
                                detection_last_id = message_id
                except Exception as e:
                    logger.error(f"Error reading detection stream: {e}")

                # Read transcriptions
                try:
                    transcription_streams = self.redis_client.xread(
                        {'stream:transcriptions': transcription_last_id},
                        count=1,
                        block=50
                    )

                    if transcription_streams:
                        for stream_name, messages in transcription_streams:
                            for message_id, message_data in messages:
                                payload = json.loads(message_data['payload'])

                                # Correlate speaker
                                speaker_event = self.correlate_speaker(payload)

                                if speaker_event:
                                    self.publish_speaker_event(speaker_event)

                                transcription_last_id = message_id
                except Exception as e:
                    logger.error(f"Error reading transcription stream: {e}")

        except KeyboardInterrupt:
            logger.info("Shutting down speaker detector...")
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            raise


def main():
    """Main entry point"""
    detector = SpeakerDetector()

    # Initialize
    detector.connect_redis()

    # Run processing loop
    detector.run()


if __name__ == '__main__':
    main()
