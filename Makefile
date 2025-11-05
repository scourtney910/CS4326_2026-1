.PHONY: help setup-jetson download-models optimize-models build run stop logs clean monitor debug profile test

help:
	@echo "Jetson Multi-Modal Edge AI Framework - Makefile Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup-jetson      - Setup Jetson environment (install dependencies)"
	@echo "  make download-models   - Download required AI models"
	@echo "  make optimize-models   - Convert models to TensorRT/ONNX for Jetson"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make build            - Build all Docker containers"
	@echo "  make run              - Start all services"
	@echo "  make stop             - Stop all services"
	@echo "  make restart          - Restart all services"
	@echo "  make clean            - Remove containers and volumes"
	@echo ""
	@echo "Development Commands:"
	@echo "  make logs             - View logs from all containers"
	@echo "  make logs-vision      - View vision container logs"
	@echo "  make logs-audio       - View audio container logs"
	@echo "  make logs-speaker     - View speaker detection logs"
	@echo "  make logs-llm         - View LLM container logs"
	@echo "  make logs-orchestrator - View orchestrator logs"
	@echo ""
	@echo "Debugging Commands:"
	@echo "  make monitor          - Monitor system resources"
	@echo "  make debug MODULE=<name> - Debug specific module"
	@echo "  make profile          - Profile system performance"
	@echo "  make test             - Run all tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-unit        - Run unit tests"

setup-jetson:
	@echo "Setting up Jetson environment..."
	bash scripts/setup_jetson.sh

download-models:
	@echo "Downloading AI models..."
	bash scripts/download_models.sh

optimize-models:
	@echo "Optimizing models for Jetson..."
	python3 scripts/optimize_models.py

build:
	@echo "Building Docker containers..."
	docker-compose build

run:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started! Use 'make logs' to view logs"

stop:
	@echo "Stopping all services..."
	docker-compose down

restart:
	@echo "Restarting all services..."
	docker-compose restart

clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	@echo "Cleanup complete"

logs:
	docker-compose logs -f

logs-vision:
	docker-compose logs -f vision

logs-audio:
	docker-compose logs -f audio

logs-speaker:
	docker-compose logs -f speaker_detection

logs-llm:
	docker-compose logs -f llm

logs-orchestrator:
	docker-compose logs -f orchestrator

monitor:
	@echo "Monitoring system resources..."
	@echo "Press Ctrl+C to stop"
	@watch -n 2 'tegrastats && echo "\n--- Docker Stats ---" && docker stats --no-stream'

debug:
	@if [ -z "$(MODULE)" ]; then \
		echo "Error: Please specify MODULE=<container_name>"; \
		echo "Example: make debug MODULE=vision"; \
		exit 1; \
	fi
	docker exec -it jetson-$(MODULE) /bin/bash

profile:
	@echo "Profiling system performance..."
	python3 scripts/profile_system.py

test:
	@echo "Running all tests..."
	pytest tests/

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/

status:
	@echo "Container Status:"
	@docker-compose ps
	@echo "\nRedis Status:"
	@docker exec jetson-redis redis-cli ping 2>/dev/null || echo "Redis not responding"
	@echo "\nGPU Status:"
	@docker exec jetson-vision nvidia-smi 2>/dev/null || echo "GPU not accessible"

shell-vision:
	docker exec -it jetson-vision /bin/bash

shell-audio:
	docker exec -it jetson-audio /bin/bash

shell-speaker:
	docker exec -it jetson-speaker-detection /bin/bash

shell-llm:
	docker exec -it jetson-llm /bin/bash

shell-orchestrator:
	docker exec -it jetson-orchestrator /bin/bash

shell-redis:
	docker exec -it jetson-redis redis-cli
