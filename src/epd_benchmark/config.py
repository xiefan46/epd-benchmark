"""
Configuration management for EPD benchmarks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass
class DatasetConfig:
    image_dir: str = "/path/to/images"
    text_data_path: str = "/path/to/conversations.jsonl"
    hf_dataset: str = "lmarena-ai/VisionArena-Chat"


@dataclass
class WorkloadConfig:
    duration_seconds: int = 600
    num_prompts: int = 100
    text_base_rate: float = 30.0
    multimodal_base_rate: float = 20.0
    text_variance: float = 0.3
    multimodal_variance: float = 1.5
    max_tokens: int = 512
    temperature: float = 0.7


@dataclass
class VLLMGPUConfig:
    encoder: int = 0
    prefill: int = 1
    decode: int = 2


@dataclass
class ElasticMMGPUConfig:
    text_gpus: int = 1
    multimodal_gpus: int = 3


@dataclass
class GPUConfig:
    total: int = 4
    vllm: VLLMGPUConfig = field(default_factory=VLLMGPUConfig)
    elasticmm: ElasticMMGPUConfig = field(default_factory=ElasticMMGPUConfig)


@dataclass
class VLLMPortConfig:
    encoder: int = 19534
    prefill: int = 19535
    decode: int = 19536
    proxy: int = 10001


@dataclass
class ElasticMMPortConfig:
    proxy: int = 10002
    service_discovery: int = 30002


@dataclass
class PortConfig:
    vllm: VLLMPortConfig = field(default_factory=VLLMPortConfig)
    elasticmm: ElasticMMPortConfig = field(default_factory=ElasticMMPortConfig)


@dataclass
class StorageConfig:
    ec_cache: str = "/tmp/ec_cache"
    logs: str = "./logs"
    results: str = "./results"


@dataclass
class BenchmarkSettings:
    warmup_seconds: int = 30
    num_runs: int = 3
    metrics_interval: float = 1.0


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    gpus: GPUConfig = field(default_factory=GPUConfig)
    ports: PortConfig = field(default_factory=PortConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            workload=WorkloadConfig(**data.get("workload", {})),
            gpus=GPUConfig(
                total=data.get("gpus", {}).get("total", 4),
                vllm=VLLMGPUConfig(**data.get("gpus", {}).get("vllm", {})),
                elasticmm=ElasticMMGPUConfig(**data.get("gpus", {}).get("elasticmm", {})),
            ),
            ports=PortConfig(
                vllm=VLLMPortConfig(**data.get("ports", {}).get("vllm", {})),
                elasticmm=ElasticMMPortConfig(**data.get("ports", {}).get("elasticmm", {})),
            ),
            storage=StorageConfig(**data.get("storage", {})),
            benchmark=BenchmarkSettings(**data.get("benchmark", {})),
        )


def load_config(config_path: str | Path) -> BenchmarkConfig:
    """Load benchmark configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return BenchmarkConfig.from_dict(data)
