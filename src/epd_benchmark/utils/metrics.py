"""
Metrics collection and analysis for EPD benchmarks.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: str
    start_time: float
    end_time: float
    latency: float
    ttft: float | None = None  # Time to first token (for streaming)
    tokens_generated: int = 0
    is_multimodal: bool = False
    success: bool = True
    error: str | None = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a benchmark run."""

    # Throughput
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    requests_per_second: float = 0.0

    # Latency (in seconds)
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0

    # TTFT (Time to First Token)
    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p99_ttft: float = 0.0

    # Token throughput
    total_tokens: int = 0
    tokens_per_second: float = 0.0

    # Breakdown by modality
    text_requests: int = 0
    multimodal_requests: int = 0
    text_avg_latency: float = 0.0
    multimodal_avg_latency: float = 0.0

    # Duration
    duration_seconds: float = 0.0


class MetricsCollector:
    """Collects and aggregates metrics during benchmark runs."""

    def __init__(self, results_dir: str | Path = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: list[RequestMetrics] = []
        self.start_time: float | None = None
        self.end_time: float | None = None

    def start(self) -> None:
        """Start metrics collection."""
        self.metrics = []
        self.start_time = time.time()
        self.end_time = None

    def stop(self) -> None:
        """Stop metrics collection."""
        self.end_time = time.time()

    def record(self, metrics: RequestMetrics) -> None:
        """Record metrics for a single request."""
        self.metrics.append(metrics)

    def aggregate(self) -> AggregatedMetrics:
        """Compute aggregated metrics from collected data."""
        if not self.metrics:
            return AggregatedMetrics()

        duration = (self.end_time or time.time()) - (self.start_time or 0)

        # Filter successful requests for latency calculations
        successful = [m for m in self.metrics if m.success]
        latencies = [m.latency for m in successful]
        ttfts = [m.ttft for m in successful if m.ttft is not None]

        # Breakdown by modality
        text_metrics = [m for m in successful if not m.is_multimodal]
        mm_metrics = [m for m in successful if m.is_multimodal]

        # Calculate percentiles
        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            return float(np.percentile(data, p))

        total_tokens = sum(m.tokens_generated for m in successful)

        return AggregatedMetrics(
            total_requests=len(self.metrics),
            successful_requests=len(successful),
            failed_requests=len(self.metrics) - len(successful),
            requests_per_second=len(self.metrics) / duration if duration > 0 else 0,
            avg_latency=np.mean(latencies) if latencies else 0.0,
            p50_latency=percentile(latencies, 50),
            p90_latency=percentile(latencies, 90),
            p95_latency=percentile(latencies, 95),
            p99_latency=percentile(latencies, 99),
            min_latency=min(latencies) if latencies else 0.0,
            max_latency=max(latencies) if latencies else 0.0,
            avg_ttft=np.mean(ttfts) if ttfts else 0.0,
            p50_ttft=percentile(ttfts, 50),
            p99_ttft=percentile(ttfts, 99),
            total_tokens=total_tokens,
            tokens_per_second=total_tokens / duration if duration > 0 else 0,
            text_requests=len(text_metrics),
            multimodal_requests=len(mm_metrics),
            text_avg_latency=np.mean([m.latency for m in text_metrics]) if text_metrics else 0.0,
            multimodal_avg_latency=np.mean([m.latency for m in mm_metrics]) if mm_metrics else 0.0,
            duration_seconds=duration,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        data = []
        for m in self.metrics:
            data.append(
                {
                    "request_id": m.request_id,
                    "start_time": m.start_time,
                    "end_time": m.end_time,
                    "latency": m.latency,
                    "ttft": m.ttft,
                    "tokens_generated": m.tokens_generated,
                    "is_multimodal": m.is_multimodal,
                    "success": m.success,
                    "error": m.error,
                }
            )
        return pd.DataFrame(data)

    def save(self, name: str, framework: str) -> Path:
        """Save metrics to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{framework}_{name}_{timestamp}"

        # Save raw metrics as JSON
        metrics_path = self.results_dir / f"{base_name}_raw.json"
        raw_data = [
            {
                "request_id": m.request_id,
                "start_time": m.start_time,
                "end_time": m.end_time,
                "latency": m.latency,
                "ttft": m.ttft,
                "tokens_generated": m.tokens_generated,
                "is_multimodal": m.is_multimodal,
                "success": m.success,
                "error": m.error,
            }
            for m in self.metrics
        ]
        with open(metrics_path, "w") as f:
            json.dump(raw_data, f, indent=2)

        # Save aggregated metrics
        agg = self.aggregate()
        summary_path = self.results_dir / f"{base_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(agg.__dict__, f, indent=2)

        # Save as CSV for easy analysis
        csv_path = self.results_dir / f"{base_name}.csv"
        self.to_dataframe().to_csv(csv_path, index=False)

        print(f"Results saved to {self.results_dir}")
        return self.results_dir


def compare_results(
    vllm_results_path: str | Path,
    elasticmm_results_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compare benchmark results between vLLM EPD and ElasticMM.

    Args:
        vllm_results_path: Path to vLLM summary JSON
        elasticmm_results_path: Path to ElasticMM summary JSON
        output_path: Optional path to save comparison results

    Returns:
        Dictionary with comparison metrics
    """
    with open(vllm_results_path) as f:
        vllm_metrics = json.load(f)

    with open(elasticmm_results_path) as f:
        elasticmm_metrics = json.load(f)

    comparison = {
        "vllm": vllm_metrics,
        "elasticmm": elasticmm_metrics,
        "comparison": {
            "throughput_ratio": (
                vllm_metrics["requests_per_second"] / elasticmm_metrics["requests_per_second"]
                if elasticmm_metrics["requests_per_second"] > 0
                else float("inf")
            ),
            "latency_ratio": (
                vllm_metrics["avg_latency"] / elasticmm_metrics["avg_latency"]
                if elasticmm_metrics["avg_latency"] > 0
                else float("inf")
            ),
            "p99_latency_ratio": (
                vllm_metrics["p99_latency"] / elasticmm_metrics["p99_latency"]
                if elasticmm_metrics["p99_latency"] > 0
                else float("inf")
            ),
            "success_rate_vllm": (
                vllm_metrics["successful_requests"] / vllm_metrics["total_requests"]
                if vllm_metrics["total_requests"] > 0
                else 0
            ),
            "success_rate_elasticmm": (
                elasticmm_metrics["successful_requests"] / elasticmm_metrics["total_requests"]
                if elasticmm_metrics["total_requests"] > 0
                else 0
            ),
        },
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)

    return comparison


def print_comparison_report(comparison: dict[str, Any]) -> None:
    """Print a formatted comparison report."""
    vllm = comparison["vllm"]
    emm = comparison["elasticmm"]
    comp = comparison["comparison"]

    print("\n" + "=" * 60)
    print("EPD Benchmark Comparison Report")
    print("=" * 60)

    print("\nThroughput:")
    print(f"  vLLM EPD:    {vllm['requests_per_second']:.2f} req/s")
    print(f"  ElasticMM:   {emm['requests_per_second']:.2f} req/s")
    print(f"  Ratio:       {comp['throughput_ratio']:.2f}x")

    print("\nAverage Latency:")
    print(f"  vLLM EPD:    {vllm['avg_latency']*1000:.2f} ms")
    print(f"  ElasticMM:   {emm['avg_latency']*1000:.2f} ms")
    print(f"  Ratio:       {comp['latency_ratio']:.2f}x")

    print("\nP99 Latency:")
    print(f"  vLLM EPD:    {vllm['p99_latency']*1000:.2f} ms")
    print(f"  ElasticMM:   {emm['p99_latency']*1000:.2f} ms")
    print(f"  Ratio:       {comp['p99_latency_ratio']:.2f}x")

    print("\nSuccess Rate:")
    print(f"  vLLM EPD:    {comp['success_rate_vllm']*100:.1f}%")
    print(f"  ElasticMM:   {comp['success_rate_elasticmm']*100:.1f}%")

    print("\nWorkload Breakdown:")
    print(f"  vLLM EPD:    {vllm['text_requests']} text, {vllm['multimodal_requests']} multimodal")
    print(f"  ElasticMM:   {emm['text_requests']} text, {emm['multimodal_requests']} multimodal")

    print("\n" + "=" * 60)
