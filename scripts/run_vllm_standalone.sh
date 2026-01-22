#!/bin/bash
# Run vLLM EPD servers standalone (for manual testing)
#
# This script is similar to vllm/examples/online_serving/disaggregated_encoder/disagg_1e1p1d_example.sh
# but uses the benchmark configuration for consistency.
#
# Usage:
#   ./scripts/run_vllm_standalone.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EPD_ROOT="$(dirname "$PROJECT_ROOT")"
VLLM_PATH="$EPD_ROOT/vllm"

# Configuration - override via environment variables
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/logs}"
mkdir -p "$LOG_PATH"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-0}"
GPU_P="${GPU_P:-1}"
GPU_D="${GPU_D:-2}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"

declare -a PIDS=()

cleanup() {
    echo "Stopping all processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "All processes stopped."
    exit 0
}

trap cleanup INT TERM

wait_for_server() {
    local port=$1
    local timeout=${2:-300}
    echo "Waiting for server on port $port..."
    timeout "$timeout" bash -c "
        until curl -s localhost:$port/health > /dev/null 2>&1; do
            sleep 1
        done" && return 0 || return 1
}

# Clear EC cache
rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

START_TIME=$(date +"%Y%m%d_%H%M%S")

echo "=================================================="
echo "Starting vLLM EPD (1E1P1D configuration)"
echo "=================================================="
echo "Model: $MODEL"
echo "Encoder: GPU $GPU_E, port $ENCODE_PORT"
echo "Prefill: GPU $GPU_P, port $PREFILL_PORT"
echo "Decode:  GPU $GPU_D, port $DECODE_PORT"
echo "Proxy:   port $PROXY_PORT"
echo "=================================================="

# Start Encoder
echo "Starting encoder..."
CUDA_VISIBLE_DEVICES="$GPU_E" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$ENCODE_PORT" \
    --gpu-memory-utilization 0.01 \
    --enforce-eager \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 128 \
    --ec-transfer-config "{
        \"ec_connector\": \"ECExampleConnector\",
        \"ec_role\": \"ec_producer\",
        \"ec_connector_extra_config\": {
            \"shared_storage_path\": \"$EC_SHARED_STORAGE_PATH\"
        }
    }" \
    > "$LOG_PATH/encoder_${START_TIME}.log" 2>&1 &
PIDS+=($!)

# Start Prefill
echo "Starting prefill..."
CUDA_VISIBLE_DEVICES="$GPU_P" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PREFILL_PORT" \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --ec-transfer-config "{
        \"ec_connector\": \"ECExampleConnector\",
        \"ec_role\": \"ec_consumer\",
        \"ec_connector_extra_config\": {
            \"shared_storage_path\": \"$EC_SHARED_STORAGE_PATH\"
        }
    }" \
    --kv-transfer-config "{
        \"kv_connector\": \"NixlConnector\",
        \"kv_role\": \"kv_producer\"
    }" \
    > "$LOG_PATH/prefill_${START_TIME}.log" 2>&1 &
PIDS+=($!)

# Start Decode
echo "Starting decode..."
CUDA_VISIBLE_DEVICES="$GPU_D" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=6000 \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$DECODE_PORT" \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --kv-transfer-config "{
        \"kv_connector\": \"NixlConnector\",
        \"kv_role\": \"kv_consumer\"
    }" \
    > "$LOG_PATH/decode_${START_TIME}.log" 2>&1 &
PIDS+=($!)

# Wait for workers
wait_for_server $ENCODE_PORT
echo "Encoder ready"

wait_for_server $PREFILL_PORT
echo "Prefill ready"

wait_for_server $DECODE_PORT
echo "Decode ready"

# Start Proxy
echo "Starting proxy..."
python "$VLLM_PATH/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    > "$LOG_PATH/proxy_${START_TIME}.log" 2>&1 &
PIDS+=($!)

wait_for_server $PROXY_PORT
echo ""
echo "=================================================="
echo "vLLM EPD is ready!"
echo "Proxy URL: http://localhost:$PROXY_PORT/v1/chat/completions"
echo "=================================================="
echo ""
echo "Logs are in: $LOG_PATH"
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for Ctrl+C
wait
