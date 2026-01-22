#!/bin/bash
# Run ElasticMM standalone (for manual testing)
#
# Usage:
#   ./scripts/run_elasticmm_standalone.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EPD_ROOT="$(dirname "$PROJECT_ROOT")"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

CONFIG="${CONFIG:-$PROJECT_ROOT/configs/benchmark_config.yaml}"
ELASTICMM_PATH="${ELASTICMM_PATH:-$EPD_ROOT/ElasticMM}"

echo "=================================================="
echo "Starting ElasticMM"
echo "=================================================="
echo "Config: $CONFIG"
echo "ElasticMM path: $ELASTICMM_PATH"
echo "=================================================="

epd-bench start-elasticmm \
    --config "$CONFIG" \
    --elasticmm-path "$ELASTICMM_PATH"
