"""
Orchestrator Container - System Coordination & GPIO Control

This module coordinates all components, correlates events across streams,
and triggers GPIO relay when hostile speaker is detected.
"""

import os
import time
import json
import redis
from typing import Dict, Optional
from collections import deque
from enum import Enum
import logging
import sqlite3
from datetime import datetime

# GPIO control (Jetson.GPIO)
try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("Jetson.GPIO not available - GPIO control disabled")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System state machine states"""
    DETECTING_PERSONS = 1
    LISTENING_FOR_SPEECH = 2
    IDENTIFYING_SPEAKER = 3
    ANALYZING_SENTIMENT = 4
    DECISION = 5
    GPIO_ACTIVE = 6


class Orchestrator:
    """Coordinates all system components and manages GPIO control"""

    def __init__(self):
        self.redis_client = None
        self.db_conn = None
        self.state = SystemState.DETECTING_PERSONS

        # Load configuration from environment
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.gpio_relay_pin = int(os.getenv('GPIO_RELAY_PIN', 7))
        self.trigger_duration = float(os.getenv('TRIGGER_DURATION', 5.0))
        self.cooldown_period = float(os.getenv('COOLDOWN_PERIOD', 2.0))
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', 0.7))
        self.max_triggers_per_minute = int(os.getenv('MAX_TRIGGERS_PER_MINUTE', 10))

        # Event correlation buffers
        self.speaker_events = deque(maxlen=50)
        self.sentiment_events = deque(maxlen=50)

        # GPIO state
        self.gpio_active = False
        self.gpio_active_until = 0
        self.last_trigger_time = 0
        self.trigger_count_minute = deque(maxlen=self.max_triggers_per_minute)

        logger.info("Orchestrator initialized")

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

    def initialize_database(self):
        """Initialize SQLite database for event logging"""
        try:
            self.db_conn = sqlite3.connect('/logs/events.db')
            cursor = self.db_conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    event_type TEXT,
                    track_id INTEGER,
                    bbox TEXT,
                    transcription TEXT,
                    classification TEXT,
                    confidence REAL,
                    action TEXT
                )
            ''')

            self.db_conn.commit()
            logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def initialize_gpio(self):
        """Initialize GPIO for relay control"""
        if not GPIO_AVAILABLE:
            logger.warning("GPIO control disabled")
            return

        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.gpio_relay_pin, GPIO.OUT, initial=GPIO.LOW)
            logger.info(f"GPIO initialized on pin {self.gpio_relay_pin}")
        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}")

    def trigger_gpio(self):
        """Activate GPIO relay"""
        if not GPIO_AVAILABLE:
            logger.info("[SIMULATION] GPIO relay triggered")
            return

        # Safety checks
        current_time = time.time()

        # Check cooldown period
        if current_time - self.last_trigger_time < self.cooldown_period:
            logger.warning("GPIO trigger ignored: cooldown period active")
            return

        # Check rate limiting
        self.trigger_count_minute = deque(
            [t for t in self.trigger_count_minute if current_time - t < 60],
            maxlen=self.max_triggers_per_minute
        )

        if len(self.trigger_count_minute) >= self.max_triggers_per_minute:
            logger.warning("GPIO trigger ignored: rate limit reached")
            return

        # Activate relay
        try:
            GPIO.output(self.gpio_relay_pin, GPIO.HIGH)
            self.gpio_active = True
            self.gpio_active_until = current_time + self.trigger_duration
            self.last_trigger_time = current_time
            self.trigger_count_minute.append(current_time)

            logger.warning(f"GPIO RELAY ACTIVATED for {self.trigger_duration}s")

            # Publish action event
            self.publish_action_event('GPIO_TRIGGERED')

        except Exception as e:
            logger.error(f"Failed to trigger GPIO: {e}")

    def deactivate_gpio(self):
        """Deactivate GPIO relay"""
        if not GPIO_AVAILABLE:
            logger.info("[SIMULATION] GPIO relay deactivated")
            self.gpio_active = False
            return

        try:
            GPIO.output(self.gpio_relay_pin, GPIO.LOW)
            self.gpio_active = False
            logger.info("GPIO relay deactivated")
        except Exception as e:
            logger.error(f"Failed to deactivate GPIO: {e}")

    def check_gpio_timeout(self):
        """Check if GPIO should be deactivated"""
        if self.gpio_active and time.time() >= self.gpio_active_until:
            self.deactivate_gpio()

    def correlate_events(self, sentiment_event: Dict) -> Optional[Dict]:
        """
        Correlate sentiment analysis with speaker identification

        Args:
            sentiment_event: Sentiment analysis event

        Returns:
            Correlated event with speaker info, or None
        """
        audio_chunk_id = sentiment_event['data']['audio_chunk_id']

        # Find matching speaker event
        for speaker_event in self.speaker_events:
            if speaker_event['data']['active_speaker']['audio_chunk_id'] == audio_chunk_id:
                return {
                    'sentiment': sentiment_event,
                    'speaker': speaker_event
                }

        return None

    def process_decision(self, correlated_event: Dict):
        """
        Make decision based on correlated events

        Args:
            correlated_event: Correlated sentiment and speaker data
        """
        sentiment_data = correlated_event['sentiment']['data']
        speaker_data = correlated_event['speaker']['data']['active_speaker']

        classification = sentiment_data['classification']
        confidence = sentiment_data['confidence']
        transcription = sentiment_data['transcription']

        logger.info(f"Decision: Classification={classification}, Confidence={confidence:.2f}")

        # Log event to database
        self.log_event(
            event_type='decision',
            track_id=speaker_data['track_id'],
            bbox=json.dumps(speaker_data['bbox']),
            transcription=transcription,
            classification=classification,
            confidence=confidence,
            action='NONE'
        )

        # Trigger GPIO if hostile and confidence is high enough
        if classification == 'HOSTILE' and confidence >= self.min_confidence:
            logger.warning(f"HOSTILE SPEECH DETECTED!")
            logger.warning(f"Speaker: track_id={speaker_data['track_id']}")
            logger.warning(f"Location: bbox={speaker_data['bbox']}")
            logger.warning(f"Transcription: '{transcription}'")

            # Trigger relay
            self.trigger_gpio()

            # Log action
            self.log_event(
                event_type='action',
                track_id=speaker_data['track_id'],
                bbox=json.dumps(speaker_data['bbox']),
                transcription=transcription,
                classification=classification,
                confidence=confidence,
                action='GPIO_TRIGGERED'
            )
        else:
            logger.info(f"SAFE speech: '{transcription}'")

    def log_event(self, event_type: str, track_id: int, bbox: str,
                  transcription: str, classification: str, confidence: float, action: str):
        """Log event to database"""
        if self.db_conn is None:
            return

        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO events (timestamp, event_type, track_id, bbox, transcription, classification, confidence, action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (time.time(), event_type, track_id, bbox, transcription, classification, confidence, action))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def publish_action_event(self, action: str):
        """Publish action event to Redis"""
        event = {
            'timestamp': time.time(),
            'component': 'orchestrator',
            'event_type': 'action',
            'data': {
                'action': action
            }
        }

        try:
            self.redis_client.xadd(
                'stream:actions',
                {'payload': json.dumps(event)}
            )
        except Exception as e:
            logger.error(f"Failed to publish action: {e}")

    def run(self):
        """Main orchestration loop"""
        logger.info("Starting orchestrator...")
        logger.info("Monitoring all event streams...")

        speaker_last_id = '0'
        sentiment_last_id = '0'

        try:
            while True:
                # Check GPIO timeout
                self.check_gpio_timeout()

                # Read speaker events
                try:
                    speaker_streams = self.redis_client.xread(
                        {'stream:speakers': speaker_last_id},
                        count=10,
                        block=50
                    )

                    if speaker_streams:
                        for stream_name, messages in speaker_streams:
                            for message_id, message_data in messages:
                                payload = json.loads(message_data['payload'])
                                self.speaker_events.append(payload)
                                speaker_last_id = message_id
                                logger.debug(f"Speaker event buffered: track_id={payload['data']['active_speaker']['track_id']}")
                except Exception as e:
                    logger.error(f"Error reading speaker stream: {e}")

                # Read sentiment events
                try:
                    sentiment_streams = self.redis_client.xread(
                        {'stream:sentiment': sentiment_last_id},
                        count=1,
                        block=50
                    )

                    if sentiment_streams:
                        for stream_name, messages in sentiment_streams:
                            for message_id, message_data in messages:
                                payload = json.loads(message_data['payload'])

                                # Correlate with speaker
                                correlated = self.correlate_events(payload)

                                if correlated:
                                    # Make decision
                                    self.process_decision(correlated)
                                else:
                                    logger.warning("Could not correlate sentiment with speaker")

                                sentiment_last_id = message_id
                except Exception as e:
                    logger.error(f"Error reading sentiment stream: {e}")

                # Small sleep to prevent busy-wait
                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Shutting down orchestrator...")
        except Exception as e:
            logger.error(f"Error in orchestration loop: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.gpio_active:
            self.deactivate_gpio()

        if GPIO_AVAILABLE:
            GPIO.cleanup()

        if self.db_conn:
            self.db_conn.close()

        logger.info("Orchestrator shut down cleanly")


def main():
    """Main entry point"""
    orchestrator = Orchestrator()

    # Initialize components
    orchestrator.connect_redis()
    orchestrator.initialize_database()
    orchestrator.initialize_gpio()

    # Run orchestration loop
    orchestrator.run()


if __name__ == '__main__':
    main()
