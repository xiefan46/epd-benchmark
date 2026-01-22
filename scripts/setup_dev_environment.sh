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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EPD_ROOT="$(dirname "$PROJECT_ROOT")"

echo "=================================================="
echo "EPD Benchmark Development Environment Setup"
echo "=================================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "EPD root (parent): $EPD_ROOT"
echo ""

# Check for required directories
if [ ! -d "$EPD_ROOT/vllm" ]; then
    echo "ERROR: vLLM directory not found at $EPD_ROOT/vllm"
    echo "Please clone vLLM repository first"
    exit 1
fi

if [ ! -d "$EPD_ROOT/ElasticMM" ]; then
    echo "ERROR: ElasticMM directory not found at $EPD_ROOT/ElasticMM"
    echo "Please clone ElasticMM repository first"
    exit 1
fi

echo "Found all required repositories."
echo ""

# Create virtual environment if it doesn't exist
VENV_PATH="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

# Install packages in editable mode
echo ""
echo "=================================================="
echo "Installing packages in editable mode..."
echo "=================================================="

# Install vLLM in editable mode
echo ""
echo "Installing vLLM (editable)..."
echo "  This may take a while due to compilation..."
pip install -e "$EPD_ROOT/vllm" --no-build-isolation 2>/dev/null || {
    echo "  Note: vLLM editable install may require additional setup."
    echo "  If this fails, try: pip install -e '$EPD_ROOT/vllm[all]'"
}

# Install ElasticMM in editable mode
echo ""
echo "Installing ElasticMM (editable)..."
pip install -e "$EPD_ROOT/ElasticMM"

# Install epd-benchmark in editable mode
echo ""
echo "Installing epd-benchmark (editable)..."
pip install -e "$PROJECT_ROOT[dev]"

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "You can now:"
echo "  1. Edit epd-benchmark code in: $PROJECT_ROOT/src"
echo "  2. Edit vLLM code in: $EPD_ROOT/vllm"
echo "  3. Edit ElasticMM code in: $EPD_ROOT/ElasticMM"
echo ""
echo "Changes will take effect immediately (no reinstall needed)."
echo ""
echo "To run benchmarks:"
echo "  epd-bench run --vllm-path $EPD_ROOT/vllm --elasticmm-path $EPD_ROOT/ElasticMM"
echo ""
