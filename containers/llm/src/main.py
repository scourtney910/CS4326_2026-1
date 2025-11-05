"""
LLM Container - Sentiment and Hostility Analysis

This module analyzes transcribed text using a local LLM to determine
if the speech contains hostile content.
"""

import os
import time
import json
import redis
from llama_cpp import Llama
from typing import Optional, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes text sentiment using local LLM"""

    def __init__(self):
        self.redis_client = None
        self.model = None

        # Load configuration from environment
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.model_path = os.getenv('MODEL_PATH', '/models/llama-3.2-1b-q4.gguf')
        self.n_ctx = int(os.getenv('N_CTX', 2048))
        self.n_threads = int(os.getenv('N_THREADS', 4))
        self.temperature = float(os.getenv('TEMPERATURE', 0.1))
        self.max_tokens = int(os.getenv('MAX_TOKENS', 50))

        # System prompt for classification
        self.system_prompt = """You are a safety classifier. Analyze speech and classify it as HOSTILE or SAFE.

HOSTILE includes:
- Threats of violence or harm
- Aggressive language or verbal abuse
- Hate speech or discriminatory language
- Intimidation or coercion

SAFE includes:
- Normal conversation
- Questions
- Neutral or positive statements
- Professional communication

Respond with ONLY one word: HOSTILE or SAFE."""

        logger.info("Sentiment Analyzer initialized")

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

    def load_model(self):
        """Load LLM model using llama.cpp"""
        try:
            logger.info(f"Loading LLM model from {self.model_path}")
            logger.info("This may take a minute...")

            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=-1,  # Use GPU for all layers
                verbose=False
            )

            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def classify_text(self, text: str) -> Dict:
        """
        Classify text as HOSTILE or SAFE

        Args:
            text: Transcribed speech text

        Returns:
            Classification result dict with classification, confidence, reasoning
        """
        try:
            # Build prompt
            prompt = f"""{self.system_prompt}

Text to analyze: "{text}"

Classification:"""

            # Generate response
            start_time = time.time()
            response = self.model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["Text", "\n\n"],
                echo=False
            )
            inference_time = time.time() - start_time

            # Extract classification
            output_text = response['choices'][0]['text'].strip().upper()

            # Parse classification (handle variations)
            if 'HOSTILE' in output_text:
                classification = 'HOSTILE'
                confidence = 0.9
            elif 'SAFE' in output_text:
                classification = 'SAFE'
                confidence = 0.9
            else:
                # Unclear classification, default to SAFE
                logger.warning(f"Unclear classification: {output_text}")
                classification = 'SAFE'
                confidence = 0.5

            result = {
                'classification': classification,
                'confidence': confidence,
                'reasoning': output_text,
                'inference_time': inference_time
            }

            logger.info(f"Classification: {classification} (confidence: {confidence:.2f}, time: {inference_time:.2f}s)")

            return result

        except Exception as e:
            logger.error(f"Classification error: {e}")
            # Default to SAFE on error
            return {
                'classification': 'SAFE',
                'confidence': 0.0,
                'reasoning': 'Error during classification',
                'inference_time': 0.0
            }

    def publish_sentiment(self, transcription_event: Dict, classification: Dict):
        """Publish sentiment analysis result to Redis"""
        event = {
            'timestamp': time.time(),
            'component': 'llm',
            'event_type': 'sentiment',
            'data': {
                'audio_chunk_id': transcription_event['data']['audio_chunk_id'],
                'transcription': transcription_event['data']['transcription'],
                'classification': classification['classification'],
                'confidence': classification['confidence'],
                'reasoning': classification['reasoning'],
                'inference_time': classification['inference_time']
            }
        }

        try:
            self.redis_client.xadd(
                'stream:sentiment',
                {'payload': json.dumps(event)}
            )
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")

    def run(self):
        """Main processing loop"""
        logger.info("Starting sentiment analysis...")
        logger.info("Listening to transcription stream...")

        last_id = '0'

        try:
            while True:
                # Read from transcription stream
                streams = self.redis_client.xread(
                    {'stream:transcriptions': last_id},
                    count=1,
                    block=1000  # Block for 1 second
                )

                if streams:
                    for stream_name, messages in streams:
                        for message_id, message_data in messages:
                            payload = json.loads(message_data['payload'])

                            # Extract transcription
                            transcription = payload['data']['transcription']
                            logger.info(f"Analyzing: '{transcription}'")

                            # Classify text
                            classification = self.classify_text(transcription)

                            # Publish result
                            self.publish_sentiment(payload, classification)

                            last_id = message_id

        except KeyboardInterrupt:
            logger.info("Shutting down sentiment analyzer...")
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            raise


def main():
    """Main entry point"""
    analyzer = SentimentAnalyzer()

    # Initialize components
    analyzer.connect_redis()
    analyzer.load_model()

    # Run processing loop
    analyzer.run()


if __name__ == '__main__':
    main()
