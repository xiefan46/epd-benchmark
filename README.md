# EPD Benchmark Suite

Unified benchmark framework for comparing **vLLM EPD** (disaggregated encoder-prefill-decode) and **ElasticMM** performance.

## Project Structure

```
epd/                              # Parent directory
├── vllm/                         # vLLM repository (editable)
├── ElasticMM/                    # ElasticMM repository (editable)
└── epd-benchmark/                # This benchmark project
    ├── src/epd_benchmark/        # Python package
    │   ├── benchmarks/           # Benchmark runner
    │   ├── launchers/            # Framework launchers
    │   └── utils/                # Utilities (request generator, metrics)
    ├── configs/                  # Configuration files
    ├── scripts/                  # Shell scripts
    ├── docker/                   # Docker files
    ├── results/                  # Benchmark results (auto-generated)
    └── logs/                     # Server logs (auto-generated)
```

## Quick Start

### Option 1: Using Docker (Recommended for Remote Servers)

Use our pre-built Docker image with Miniconda:

```bash
# Pull the image
docker pull fxie46/epd-benchmark:latest

# Run container (RunPod, etc.)
# Image: fxie46/epd-benchmark:latest

# Inside the container, run setup
./scripts/setup_remote_server.sh
```

### Option 2: Local Setup with Conda

```bash
# Clone the repository
git clone https://github.com/xiefan46/epd-benchmark.git
cd epd-benchmark

# Run setup script (uses Conda)
./scripts/setup_dev_environment.sh

# Activate the environment
conda activate epd
```

### Option 3: Using environment.yaml

```bash
# Create environment from file
conda env create -f environment.yaml

# Activate
conda activate epd

# Install in editable mode
pip install -e .
```

## Docker Image

### Using the Pre-built Image

The Docker image `fxie46/epd-benchmark` includes:
- Base: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Miniconda with `epd` environment pre-configured
- Common dependencies pre-installed

```bash
# On RunPod or similar platforms
# Set image to: fxie46/epd-benchmark:latest

# Inside the container
conda activate epd
cd /workspace
git clone https://github.com/xiefan46/epd-benchmark.git
./epd-benchmark/scripts/setup_remote_server.sh
```

### Building Your Own Image

```bash
cd docker
./build_and_push.sh latest
```

## Remote Server Setup (RunPod, etc.)

### One-liner Setup

```bash
curl -sSL https://raw.githubusercontent.com/xiefan46/epd-benchmark/main/scripts/setup_remote_server.sh | bash
```

### Manual Setup

```bash
# 1. Activate conda environment
conda activate epd

# 2. Clone repositories
cd /workspace
git clone https://github.com/vllm-project/vllm.git
git clone https://github.com/xiefan46/ElasticMM.git  # Your ElasticMM fork
git clone https://github.com/xiefan46/epd-benchmark.git

# 3. Install in editable mode
pip install -e vllm
pip install -e ElasticMM
pip install -e epd-benchmark

# 4. Run benchmark
cd epd-benchmark
epd-bench run --vllm-path ../vllm --elasticmm-path ../ElasticMM
```

## Configuration

Edit `configs/benchmark_config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"

dataset:
  image_dir: "/path/to/your/images"
  text_data_path: "/path/to/conversations.jsonl"

gpus:
  vllm:
    encoder: 0
    prefill: 1
    decode: 2
  elasticmm:
    text_gpus: 1
    multimodal_gpus: 3

workload:
  duration_seconds: 600
  text_base_rate: 30.0
  multimodal_base_rate: 20.0
```

## Usage

### CLI Commands

```bash
# Run full benchmark comparison
epd-bench run --vllm-path ../vllm --elasticmm-path ../ElasticMM

# Run benchmark on specific framework
epd-bench run --framework vllm ...
epd-bench run --framework elasticmm ...

# Start servers for manual testing
epd-bench start-vllm --vllm-path ../vllm
epd-bench start-elasticmm --elasticmm-path ../ElasticMM

# Generate workload against a running server
epd-bench workload --url http://localhost:10001/v1/chat/completions --duration 60

# Compare two result files
epd-bench compare results/vllm_run_0_summary.json results/elasticmm_run_0_summary.json
```

### Standalone Server Scripts

```bash
# Start vLLM EPD servers (1E1P1D configuration)
./scripts/run_vllm_standalone.sh

# Start ElasticMM system
./scripts/run_elasticmm_standalone.sh
```

### Python API

```python
import asyncio
from epd_benchmark import load_config, DynamicRequestGenerator
from epd_benchmark.benchmarks import BenchmarkRunner

# Load configuration
config = load_config("configs/benchmark_config.yaml")

# Run benchmark
runner = BenchmarkRunner(
    config=config,
    vllm_path="../vllm",
    elasticmm_path="../ElasticMM"
)

# Run comparison
results = asyncio.run(runner.run_comparison())

# Or use the request generator directly
generator = DynamicRequestGenerator(
    proxy_url="http://localhost:10001/v1/chat/completions",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    image_dir="/path/to/images"
)

stats = asyncio.run(generator.run(
    duration_seconds=60,
    text_base_rate=10.0,
    multimodal_base_rate=5.0
))
```

## Managing Three Repositories

### Directory Layout

```
/workspace/                # Or ~/epd/
├── vllm/                  # git clone of vLLM
├── ElasticMM/             # git clone of ElasticMM
└── epd-benchmark/         # This project
```

### Editable Installs

All packages are installed in editable mode:
- `pip install -e vllm`
- `pip install -e ElasticMM`
- `pip install -e epd-benchmark`

This means:
1. Modify source code in any repo
2. Changes take effect immediately
3. Debug across all three codebases
4. Each repo maintains its own git history

## Benchmark Metrics

The benchmark collects and compares:

| Metric | Description |
|--------|-------------|
| Throughput | Requests per second |
| Latency | Average, P50, P90, P95, P99 |
| TTFT | Time to First Token |
| Token Throughput | Tokens per second |
| Success Rate | Successful vs failed requests |

Results are saved to `results/` in JSON and CSV formats.

## Troubleshooting

### vLLM won't start
- Check GPU availability: `nvidia-smi`
- Check logs: `tail -f logs/encoder_*.log`
- Ensure model is downloaded

### ElasticMM won't start
- Stop existing Ray: `ray stop`
- Check port availability
- Review logs

### Conda environment issues
```bash
# Recreate environment
conda env remove -n epd
conda env create -f environment.yaml
```

### Import errors
```bash
# Reinstall in editable mode
conda activate epd
pip install -e .
```
