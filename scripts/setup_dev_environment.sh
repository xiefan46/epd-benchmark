#!/bin/bash
# Setup script for EPD Benchmark development environment
#
# This script sets up a development environment where you can:
# 1. Modify the benchmark project (epd-benchmark)
# 2. Modify vLLM source code
# 3. Modify ElasticMM source code
#
# All three packages are installed in "editable" mode, so changes
# to source code take effect immediately without reinstalling.
#
# Uses Conda for environment management.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EPD_ROOT="$(dirname "$PROJECT_ROOT")"

ENV_NAME="${ENV_NAME:-epd}"

echo "=================================================="
echo "EPD Benchmark Development Environment Setup"
echo "=================================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "EPD root (parent): $EPD_ROOT"
echo "Conda environment: $ENV_NAME"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found. Please install Miniconda or Anaconda first."
    echo ""
    echo "Install Miniconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    exit 1
fi

# Initialize conda for current shell
eval "$(conda shell.bash hook)"

# Check for required directories
if [ ! -d "$EPD_ROOT/vllm" ]; then
    echo "WARNING: vLLM directory not found at $EPD_ROOT/vllm"
    echo "You can clone it later: git clone https://github.com/vllm-project/vllm.git $EPD_ROOT/vllm"
fi

if [ ! -d "$EPD_ROOT/ElasticMM" ]; then
    echo "WARNING: ElasticMM directory not found at $EPD_ROOT/ElasticMM"
    echo "Please clone the ElasticMM repository"
fi

# Create conda environment if it doesn't exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '$ENV_NAME' already exists"
    read -p "Recreate environment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
        echo "Creating new conda environment..."
        conda create -n "$ENV_NAME" python=3.11 -y
    fi
else
    echo "Creating conda environment '$ENV_NAME' with Python 3.11..."
    conda create -n "$ENV_NAME" python=3.11 -y
fi

echo ""
echo "Activating conda environment..."
conda activate "$ENV_NAME"

echo "Upgrading pip..."
pip install --upgrade pip

# Install base dependencies
echo ""
echo "Installing base dependencies..."
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

# Install packages in editable mode
echo ""
echo "=================================================="
echo "Installing packages in editable mode..."
echo "=================================================="

# Install vLLM in editable mode
if [ -d "$EPD_ROOT/vllm" ]; then
    echo ""
    echo "Installing vLLM (editable)..."
    echo "  This may take a while due to compilation..."
    pip install -e "$EPD_ROOT/vllm" --no-build-isolation 2>/dev/null || {
        echo "  Note: vLLM editable install may require additional setup."
        echo "  Trying alternative installation..."
        pip install -e "$EPD_ROOT/vllm" || echo "  vLLM installation failed. Install manually."
    }
else
    echo "Skipping vLLM (not found)"
fi

# Install ElasticMM in editable mode
if [ -d "$EPD_ROOT/ElasticMM" ]; then
    echo ""
    echo "Installing ElasticMM (editable)..."
    pip install -e "$EPD_ROOT/ElasticMM"
else
    echo "Skipping ElasticMM (not found)"
fi

# Install epd-benchmark in editable mode
echo ""
echo "Installing epd-benchmark (editable)..."
pip install -e "$PROJECT_ROOT"

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "You can now:"
echo "  1. Edit epd-benchmark code in: $PROJECT_ROOT/src"
if [ -d "$EPD_ROOT/vllm" ]; then
    echo "  2. Edit vLLM code in: $EPD_ROOT/vllm"
fi
if [ -d "$EPD_ROOT/ElasticMM" ]; then
    echo "  3. Edit ElasticMM code in: $EPD_ROOT/ElasticMM"
fi
echo ""
echo "Changes will take effect immediately (no reinstall needed)."
echo ""
echo "To run benchmarks:"
echo "  epd-bench run --vllm-path $EPD_ROOT/vllm --elasticmm-path $EPD_ROOT/ElasticMM"
echo ""
