"""
Vision Container - Person Detection and Tracking with YOLO

This module captures video from camera, detects persons using YOLO,
tracks them across frames, and publishes detection events to Redis.
"""

import os
import time
import json
import redis
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionProcessor:
    """Handles person detection and tracking using YOLO"""

    def __init__(self):
        self.redis_client = None
        self.model = None
        self.camera = None
        self.frame_id = 0

        # Load configuration from environment
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.model_path = os.getenv('MODEL_PATH', '/models/yolov8n.pt')
        self.camera_index = int(os.getenv('CAMERA_INDEX', 0))
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
        self.iou_threshold = float(os.getenv('IOU_THRESHOLD', 0.45))

        logger.info("Vision Processor initialized")

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
        """Load YOLO model (TensorRT engine if available, otherwise PyTorch)"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            logger.info(f"Initializing camera {self.camera_index}")
            self.camera = cv2.VideoCapture(self.camera_index)

            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")

            # Set camera properties for optimal performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise

    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame using YOLO

        Args:
            frame: Input image frame

        Returns:
            List of detection dictionaries with bbox, confidence, etc.
        """
        results = self.model.track(
            frame,
            classes=[0],  # Person class in COCO dataset
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            persist=True,  # Enable tracking
            verbose=False
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())

                # Get track ID if available
                track_id = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else -1

                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                detection = {
                    'track_id': track_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center_x), float(center_y)],
                    'confidence': conf
                }
                detections.append(detection)

        return detections

    def publish_detections(self, detections: List[Dict], timestamp: float):
        """Publish detection events to Redis stream"""
        event = {
            'timestamp': timestamp,
            'frame_id': self.frame_id,
            'component': 'vision',
            'event_type': 'detection',
            'sequence_id': self.frame_id,
            'data': {
                'detections': detections,
                'num_persons': len(detections)
            }
        }

        try:
            self.redis_client.xadd(
                'stream:detections',
                {'payload': json.dumps(event)}
            )
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")

    def run(self):
        """Main processing loop"""
        logger.info("Starting vision processing loop...")

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                timestamp = time.time()
                self.frame_id += 1

                # Detect persons in frame
                detections = self.detect_persons(frame)

                # Publish to Redis
                if detections:
                    self.publish_detections(detections, timestamp)
                    logger.debug(f"Frame {self.frame_id}: Detected {len(detections)} person(s)")

                # Optional: Save frame to shared volume for speaker detection
                # (Implement if speaker detection needs video frames)

        except KeyboardInterrupt:
            logger.info("Shutting down vision processor...")
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.camera is not None:
            self.camera.release()
        logger.info("Vision processor shut down cleanly")


def main():
    """Main entry point"""
    processor = VisionProcessor()

    # Initialize components
    processor.connect_redis()
    processor.load_model()
    processor.initialize_camera()

    # Run processing loop
    processor.run()


if __name__ == '__main__':
    main()
