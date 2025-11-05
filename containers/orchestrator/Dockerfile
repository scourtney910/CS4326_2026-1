# Orchestrator Container - System Coordination & GPIO Control
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install \
    redis \
    Jetson.GPIO \
    pyyaml \
    python-dotenv \
    asyncio \
    aioredis

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/

# Install additional requirements
RUN pip3 install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create non-root user (but note: GPIO access requires specific permissions)
RUN useradd -m -u 1000 orchestrator && chown -R orchestrator:orchestrator /app

# Note: Container runs as root for GPIO access, or add user to gpio group
# USER orchestrator

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run the application
CMD ["python3", "src/main.py"]
