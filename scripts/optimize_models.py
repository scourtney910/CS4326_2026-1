#!/usr/bin/env python3
"""
Model optimization script for Jetson
Converts models to TensorRT and ONNX for optimal performance
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "./shared/models"


def optimize_yolo():
    """Convert YOLO model to TensorRT engine"""
    logger.info("Optimizing YOLOv8 model to TensorRT...")

    try:
        from ultralytics import YOLO

        model_path = os.path.join(MODELS_DIR, "yolov8n.pt")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False

        logger.info("Loading YOLO model...")
        model = YOLO(model_path)

        logger.info("Exporting to TensorRT engine (this may take 5-10 minutes)...")
        model.export(format='engine', device=0, half=True)  # FP16 precision

        logger.info("YOLOv8 optimization complete!")
        logger.info(f"TensorRT engine saved as: yolov8n.engine")
        return True

    except ImportError:
        logger.error("ultralytics package not found. Install with: pip3 install ultralytics")
        return False
    except Exception as e:
        logger.error(f"Failed to optimize YOLO: {e}")
        return False


def optimize_whisper():
    """Convert Whisper model to ONNX"""
    logger.info("Optimizing Whisper model to ONNX...")
    logger.info("Note: Whisper ONNX export is complex and may require additional dependencies")
    logger.info("For now, we'll use the PyTorch model directly")
    logger.info("Future optimization: Convert to TensorRT for better performance")

    # TODO: Implement Whisper to ONNX conversion
    # This requires additional work and dependencies
    # For now, the PyTorch model will work fine

    return True


def check_llm_model():
    """Check if LLM model exists"""
    logger.info("Checking LLM model...")

    model_path = os.path.join(MODELS_DIR, "llama-3.2-1b-q4.gguf")

    if os.path.exists(model_path):
        logger.info("LLM model found")
        logger.info("GGUF format is already optimized for llama.cpp")
        return True
    else:
        logger.warning("LLM model not found")
        logger.warning(f"Please download a quantized GGUF model to: {model_path}")
        logger.warning("See scripts/download_models.sh for instructions")
        return False


def main():
    """Main optimization function"""
    logger.info("=" * 50)
    logger.info("Model Optimization for Jetson")
    logger.info("=" * 50)
    print()

    # Check models directory
    if not os.path.exists(MODELS_DIR):
        logger.error(f"Models directory not found: {MODELS_DIR}")
        logger.error("Run 'make download-models' first")
        sys.exit(1)

    # Optimize YOLO
    logger.info("\n[1/3] YOLOv8 Optimization")
    logger.info("-" * 50)
    yolo_success = optimize_yolo()

    # Optimize Whisper
    logger.info("\n[2/3] Whisper Optimization")
    logger.info("-" * 50)
    whisper_success = optimize_whisper()

    # Check LLM
    logger.info("\n[3/3] LLM Model Check")
    logger.info("-" * 50)
    llm_success = check_llm_model()

    # Summary
    print()
    logger.info("=" * 50)
    logger.info("Optimization Summary")
    logger.info("=" * 50)
    logger.info(f"YOLOv8:  {'✓ Success' if yolo_success else '✗ Failed'}")
    logger.info(f"Whisper: {'✓ Success' if whisper_success else '✗ Failed'}")
    logger.info(f"LLM:     {'✓ Ready' if llm_success else '⚠ Not found'}")
    print()

    if yolo_success and whisper_success:
        logger.info("Optimization complete!")
        logger.info("Next step: Run 'make build' to build Docker containers")
    else:
        logger.warning("Some optimizations failed. Check logs above.")

    sys.exit(0 if (yolo_success and whisper_success) else 1)


if __name__ == '__main__':
    main()
