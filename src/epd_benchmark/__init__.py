"""
EPD Benchmark Suite

A unified benchmark framework for comparing vLLM EPD and ElasticMM
in disaggregated encoder-prefill-decode scenarios.
"""

__version__ = "0.1.0"

from epd_benchmark.config import BenchmarkConfig, load_config
from epd_benchmark.utils.request_generator import DynamicRequestGenerator
from epd_benchmark.utils.metrics import MetricsCollector

__all__ = [
    "BenchmarkConfig",
    "load_config",
    "DynamicRequestGenerator",
    "MetricsCollector",
]
