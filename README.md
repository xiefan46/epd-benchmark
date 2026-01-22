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
    ├── results/                  # Benchmark results (auto-generated)
    └── logs/                     # Server logs (auto-generated)
```

## Quick Start

### 1. Setup Development Environment

This setup allows you to modify code in all three repositories:

```bash
# Run the setup script
./scripts/setup_dev_environment.sh

# Activate the virtual environment
source .venv/bin/activate
```

The setup installs all three packages in **editable mode** (`pip install -e`), meaning:
- Changes to `epd-benchmark/src` take effect immediately
- Changes to `vllm/` take effect immediately
- Changes to `ElasticMM/` take effect immediately

### 2. Configure the Benchmark

Edit `configs/benchmark_config.yaml` to match your environment:

```yaml
# Model to use
model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"

# Dataset paths
dataset:
  image_dir: "/path/to/your/images"
  text_data_path: "/path/to/conversations.jsonl"

# GPU assignment
gpus:
  vllm:
    encoder: 0
    prefill: 1
    decode: 2
  elasticmm:
    text_gpus: 1
    multimodal_gpus: 3
```

### 3. Run Benchmarks

```bash
# Run comparison on both frameworks
./scripts/run_benchmark.sh

# Or run individually
./scripts/run_benchmark.sh --framework vllm
./scripts/run_benchmark.sh --framework elasticmm
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

For manual testing or debugging:

```bash
# Start vLLM EPD servers (1E1P1D configuration)
./scripts/run_vllm_standalone.sh

# Start ElasticMM system
./scripts/run_elasticmm_standalone.sh
```

### Python API

```python
import asyncio
from epd_benchmark import load_config, DynamicRequestGenerator, MetricsCollector
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

The recommended workflow for managing vLLM, ElasticMM, and this benchmark project:

### Directory Layout

```
epd/
├── vllm/           # git clone of vLLM
├── ElasticMM/      # git clone of ElasticMM
└── epd-benchmark/  # This project
```

### Git Management

Each repository is managed independently:

```bash
# Work on vLLM
cd epd/vllm
git checkout -b my-feature
# ... make changes ...
git commit -m "My vLLM changes"

# Work on ElasticMM
cd epd/ElasticMM
git checkout -b my-feature
# ... make changes ...
git commit -m "My ElasticMM changes"

# Work on benchmark
cd epd/epd-benchmark
git checkout -b my-feature
# ... make changes ...
git commit -m "My benchmark changes"
```

### Editable Installs

After running `setup_dev_environment.sh`, all three packages are installed in editable mode:

- **vLLM**: `pip install -e ../vllm`
- **ElasticMM**: `pip install -e ../ElasticMM`
- **epd-benchmark**: `pip install -e .`

This means:
1. You can modify source code in any of the three repos
2. Changes take effect immediately (no reinstall needed)
3. You can set breakpoints and debug across all three codebases
4. Each repo maintains its own git history

### Recommended IDE Setup

Configure your IDE to treat all three directories as a multi-root workspace:

**VSCode** (`.vscode/settings.json`):
```json
{
  "python.analysis.extraPaths": [
    "../vllm",
    "../ElasticMM",
    "./src"
  ]
}
```

## Benchmark Metrics

The benchmark collects and compares:

- **Throughput**: Requests per second
- **Latency**: Average, P50, P90, P95, P99
- **Time to First Token (TTFT)**: For streaming requests
- **Token Throughput**: Tokens per second
- **Success Rate**: Successful vs failed requests
- **Workload Breakdown**: Text vs multimodal requests

Results are saved to `results/` in JSON and CSV formats for further analysis.

## Configuration Options

See `configs/benchmark_config.yaml` for all available options:

| Option | Description | Default |
|--------|-------------|---------|
| `model.name` | Model to benchmark | Qwen/Qwen2.5-VL-3B-Instruct |
| `workload.duration_seconds` | Benchmark duration | 600 |
| `workload.text_base_rate` | Text requests/second | 30.0 |
| `workload.multimodal_base_rate` | MM requests/second | 20.0 |
| `benchmark.warmup_seconds` | Warmup period | 30 |
| `benchmark.num_runs` | Number of runs | 3 |

## Troubleshooting

### vLLM won't start
- Check GPU availability: `nvidia-smi`
- Check logs in `logs/encoder_*.log`, `logs/prefill_*.log`, `logs/decode_*.log`
- Ensure the model is downloaded

### ElasticMM won't start
- Ensure Ray is not already running: `ray stop`
- Check if ports are available
- Review ElasticMM logs

### Import errors
- Ensure you've run `setup_dev_environment.sh`
- Ensure the virtual environment is activated
- Try reinstalling: `pip install -e .`
