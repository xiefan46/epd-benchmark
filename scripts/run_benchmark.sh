#!/bin/bash
# Run EPD benchmark comparison
#
# Usage:
#   ./scripts/run_benchmark.sh [options]
#
# Options:
#   --framework vllm|elasticmm|both  (default: both)
#   --config PATH                    (default: configs/benchmark_config.yaml)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EPD_ROOT="$(dirname "$PROJECT_ROOT")"

# Default values
FRAMEWORK="both"
CONFIG="$PROJECT_ROOT/configs/benchmark_config.yaml"
VLLM_PATH="$EPD_ROOT/vllm"
ELASTICMM_PATH="$EPD_ROOT/ElasticMM"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --framework|-f)
            FRAMEWORK="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --vllm-path)
            VLLM_PATH="$2"
            shift 2
            ;;
        --elasticmm-path)
            ELASTICMM_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "EPD Benchmark"
echo "=================================================="
echo "Framework: $FRAMEWORK"
echo "Config: $CONFIG"
echo "vLLM path: $VLLM_PATH"
echo "ElasticMM path: $ELASTICMM_PATH"
echo "=================================================="
echo ""

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Run benchmark
epd-bench run \
    --config "$CONFIG" \
    --vllm-path "$VLLM_PATH" \
    --elasticmm-path "$ELASTICMM_PATH" \
    --framework "$FRAMEWORK"
