#!/bin/bash
# Setup script for remote server (RunPod, etc.)
#
# This script sets up the complete EPD benchmark environment on a remote server.
# Assumes the Docker image with Miniconda is already running.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/xiefan46/epd-benchmark/main/scripts/setup_remote_server.sh | bash
#   # Or after cloning:
#   ./scripts/setup_remote_server.sh

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
cd "$WORKSPACE"

echo "=================================================="
echo "EPD Benchmark Remote Server Setup"
echo "=================================================="
echo "Workspace: $WORKSPACE"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found. Please use the Docker image with Miniconda."
    echo "       xiefan46/epd-benchmark:latest"
    exit 1
fi

# Initialize conda for current shell
eval "$(conda shell.bash hook)"

# Create or activate conda environment
if conda env list | grep -q "^epd "; then
    echo "Conda environment 'epd' already exists, activating..."
    conda activate epd
else
    echo "Creating conda environment 'epd'..."
    conda create -n epd python=3.11 -y
    conda activate epd
fi

echo ""
echo "Installing base dependencies..."
pip install --upgrade pip
pip install \
    aiohttp>=3.9.0 \
    httpx>=0.25.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    pyyaml>=6.0 \
    rich>=13.0.0 \
    click>=8.0.0 \
    Pillow>=10.0.0 \
    matplotlib>=3.7.0 \
    pytest>=7.0.0 \
    pytest-asyncio>=0.21.0 \
    ray>=2.8.0 \
    uvicorn>=0.24.0 \
    fastapi>=0.104.0

# Clone repositories if not present
echo ""
echo "Setting up repositories..."

# Clone vLLM
if [ ! -d "$WORKSPACE/vllm" ]; then
    echo "Cloning vLLM..."
    git clone https://github.com/vllm-project/vllm.git "$WORKSPACE/vllm"
else
    echo "vLLM already exists, pulling latest..."
    cd "$WORKSPACE/vllm" && git pull || true
    cd "$WORKSPACE"
fi

# Clone ElasticMM
if [ ! -d "$WORKSPACE/ElasticMM" ]; then
    echo "Cloning ElasticMM..."
    # Replace with your ElasticMM repo URL
    git clone https://github.com/xiefan46/ElasticMM.git "$WORKSPACE/ElasticMM" 2>/dev/null || {
        echo "Note: ElasticMM repo not found. Please clone manually:"
        echo "  git clone <your-elasticmm-repo> $WORKSPACE/ElasticMM"
    }
else
    echo "ElasticMM already exists"
fi

# Clone epd-benchmark
if [ ! -d "$WORKSPACE/epd-benchmark" ]; then
    echo "Cloning epd-benchmark..."
    git clone https://github.com/xiefan46/epd-benchmark.git "$WORKSPACE/epd-benchmark"
else
    echo "epd-benchmark already exists, pulling latest..."
    cd "$WORKSPACE/epd-benchmark" && git pull || true
    cd "$WORKSPACE"
fi

# Install packages in editable mode
echo ""
echo "Installing packages in editable mode..."

# Install vLLM
echo "Installing vLLM (this may take a while)..."
cd "$WORKSPACE/vllm"
pip install -e . --no-build-isolation 2>/dev/null || {
    echo "vLLM standard install failed, trying with build isolation..."
    pip install -e .
}

# Install ElasticMM if present
if [ -d "$WORKSPACE/ElasticMM" ]; then
    echo "Installing ElasticMM..."
    cd "$WORKSPACE/ElasticMM"
    pip install -e .
fi

# Install epd-benchmark
echo "Installing epd-benchmark..."
cd "$WORKSPACE/epd-benchmark"
pip install -e .

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the environment:"
echo "  conda activate epd"
echo ""
echo "Directory structure:"
echo "  $WORKSPACE/vllm         - vLLM (editable)"
echo "  $WORKSPACE/ElasticMM    - ElasticMM (editable)"
echo "  $WORKSPACE/epd-benchmark - Benchmark suite (editable)"
echo ""
echo "Quick start:"
echo "  cd $WORKSPACE/epd-benchmark"
echo "  # Edit configs/benchmark_config.yaml"
echo "  epd-bench run --vllm-path ../vllm --elasticmm-path ../ElasticMM"
echo ""
echo "Or run standalone tests:"
echo "  ./scripts/run_vllm_standalone.sh"
echo ""
