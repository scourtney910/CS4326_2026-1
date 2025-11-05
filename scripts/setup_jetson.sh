#!/bin/bash

# Setup script for Jetson Orin Nano
# This script installs all necessary dependencies and configures the system

set -e

echo "======================================"
echo "Jetson Multi-Modal Framework Setup"
echo "======================================"
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: This script is designed for NVIDIA Jetson devices"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker (if not already installed)
if ! command -v docker &> /dev/null; then
    echo "[2/8] Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
else
    echo "[2/8] Docker already installed"
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "[3/8] Installing Docker Compose..."
    sudo apt-get install -y docker-compose
else
    echo "[3/8] Docker Compose already installed"
fi

# Install NVIDIA Container Runtime (should be pre-installed on JetPack)
echo "[4/8] Configuring NVIDIA Container Runtime..."
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Install system dependencies
echo "[5/8] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    git \
    build-essential \
    cmake \
    wget \
    curl \
    nano \
    htop \
    v4l-utils \
    alsa-utils

# Install Python packages for scripts
echo "[6/8] Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    pyyaml \
    requests \
    tqdm

# Create directory structure
echo "[7/8] Creating directory structure..."
mkdir -p shared/models
mkdir -p shared/data/{video,audio}
mkdir -p shared/config
mkdir -p shared/logs

# Set permissions
echo "[8/8] Setting permissions..."
chmod +x scripts/*.sh
chmod +x scripts/*.py

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Log out and back in for Docker group changes to take effect"
echo "2. Run 'make download-models' to download AI models"
echo "3. Run 'make optimize-models' to optimize models for Jetson"
echo "4. Run 'make build' to build Docker containers"
echo "5. Run 'make run' to start the system"
echo ""
