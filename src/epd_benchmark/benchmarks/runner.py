"""
Unified Benchmark Runner

Runs benchmarks on both vLLM EPD and ElasticMM with identical workloads
for fair comparison.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from epd_benchmark.config import BenchmarkConfig, load_config
from epd_benchmark.launchers.vllm_launcher import VLLMLauncher
from epd_benchmark.launchers.elasticmm_launcher import ElasticMMLauncher
from epd_benchmark.utils.request_generator import DynamicRequestGenerator, RequestStats
from epd_benchmark.utils.metrics import (
    MetricsCollector,
    RequestMetrics,
    AggregatedMetrics,
    compare_results,
    print_comparison_report,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    framework: str
    run_id: int
    stats: RequestStats
    metrics: AggregatedMetrics
    results_path: Path


class BenchmarkRunner:
    """
    Unified benchmark runner for vLLM EPD vs ElasticMM comparison.

    Manages the complete benchmark lifecycle:
    1. Start framework servers
    2. Warmup period
    3. Run workload
    4. Collect metrics
    5. Stop servers
    6. Generate comparison report
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        vllm_path: str | Path,
        elasticmm_path: str | Path,
    ):
        self.config = config
        self.vllm_path = Path(vllm_path)
        self.elasticmm_path = Path(elasticmm_path)

        self.results_dir = Path(config.storage.results)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config.storage.logs)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def _run_workload(
        self,
        proxy_url: str,
        framework: str,
        run_id: int,
    ) -> tuple[RequestStats, MetricsCollector]:
        """Run the workload against a framework and collect metrics."""
        print(f"\nRunning workload on {framework} (run {run_id + 1})...")

        # Create request generator
        generator = DynamicRequestGenerator(
            proxy_url=proxy_url,
            model_name=self.config.model.name,
            image_dir=self.config.dataset.image_dir,
            text_data_path=self.config.dataset.text_data_path,
            max_tokens=self.config.workload.max_tokens,
            temperature=self.config.workload.temperature,
        )

        # Create metrics collector
        collector = MetricsCollector(results_dir=self.results_dir)

        # Progress callback
        def on_progress(stats: RequestStats):
            success_rate = (
                stats.successful_requests / stats.total_requests_sent * 100
                if stats.total_requests_sent > 0
                else 0
            )
            print(
                f"  Progress: {stats.total_requests_sent} requests, "
                f"{stats.successful_requests} successful ({success_rate:.1f}%), "
                f"avg latency: {stats.avg_latency*1000:.1f}ms"
            )

        # Warmup period
        if self.config.benchmark.warmup_seconds > 0:
            print(f"  Warmup for {self.config.benchmark.warmup_seconds}s...")
            warmup_generator = DynamicRequestGenerator(
                proxy_url=proxy_url,
                model_name=self.config.model.name,
                image_dir=self.config.dataset.image_dir,
                text_data_path=self.config.dataset.text_data_path,
                max_tokens=self.config.workload.max_tokens,
                temperature=self.config.workload.temperature,
            )
            await warmup_generator.run(
                duration_seconds=self.config.benchmark.warmup_seconds,
                text_base_rate=self.config.workload.text_base_rate / 2,
                multimodal_base_rate=self.config.workload.multimodal_base_rate / 2,
                text_variance=self.config.workload.text_variance,
                multimodal_variance=self.config.workload.multimodal_variance,
            )
            print("  Warmup complete")

        # Run actual benchmark
        collector.start()
        stats = await generator.run(
            duration_seconds=self.config.workload.duration_seconds,
            text_base_rate=self.config.workload.text_base_rate,
            multimodal_base_rate=self.config.workload.multimodal_base_rate,
            text_variance=self.config.workload.text_variance,
            multimodal_variance=self.config.workload.multimodal_variance,
            on_progress=on_progress,
        )
        collector.stop()

        # Convert stats to metrics
        for i, latency in enumerate(stats.latencies):
            is_mm = i >= stats.text_requests  # Approximation
            collector.record(
                RequestMetrics(
                    request_id=f"{framework}_{run_id}_{i}",
                    start_time=collector.start_time + i * 0.1,  # Approximation
                    end_time=collector.start_time + i * 0.1 + latency,
                    latency=latency,
                    is_multimodal=is_mm,
                    success=True,
                )
            )

        return stats, collector

    async def run_vllm(self, run_id: int = 0) -> BenchmarkResult:
        """Run benchmark on vLLM EPD."""
        print("\n" + "=" * 60)
        print(f"Starting vLLM EPD Benchmark (Run {run_id + 1})")
        print("=" * 60)

        launcher = VLLMLauncher(
            config=self.config, vllm_path=self.vllm_path, log_dir=self.log_dir
        )

        try:
            proxy_url = await launcher.start()
            stats, collector = await self._run_workload(proxy_url, "vllm", run_id)

            # Save results
            collector.save(f"run_{run_id}", "vllm")
            metrics = collector.aggregate()

            return BenchmarkResult(
                framework="vllm",
                run_id=run_id,
                stats=stats,
                metrics=metrics,
                results_path=self.results_dir,
            )
        finally:
            launcher.stop()

    async def run_elasticmm(self, run_id: int = 0) -> BenchmarkResult:
        """Run benchmark on ElasticMM."""
        print("\n" + "=" * 60)
        print(f"Starting ElasticMM Benchmark (Run {run_id + 1})")
        print("=" * 60)

        launcher = ElasticMMLauncher(
            config=self.config, elasticmm_path=self.elasticmm_path
        )

        try:
            proxy_url = await launcher.start()
            stats, collector = await self._run_workload(proxy_url, "elasticmm", run_id)

            # Save results
            collector.save(f"run_{run_id}", "elasticmm")
            metrics = collector.aggregate()

            return BenchmarkResult(
                framework="elasticmm",
                run_id=run_id,
                stats=stats,
                metrics=metrics,
                results_path=self.results_dir,
            )
        finally:
            await launcher.stop()

    async def run_comparison(
        self, frameworks: list[Literal["vllm", "elasticmm"]] | None = None
    ) -> dict:
        """
        Run complete comparison benchmark.

        Args:
            frameworks: List of frameworks to benchmark. Defaults to both.

        Returns:
            Dictionary with comparison results
        """
        if frameworks is None:
            frameworks = ["vllm", "elasticmm"]

        num_runs = self.config.benchmark.num_runs
        results = {"vllm": [], "elasticmm": []}

        # Run benchmarks for each framework
        for framework in frameworks:
            for run_id in range(num_runs):
                if framework == "vllm":
                    result = await self.run_vllm(run_id)
                else:
                    result = await self.run_elasticmm(run_id)
                results[framework].append(result)

        # Print summary
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)

        for framework in frameworks:
            if results[framework]:
                avg_latency = sum(r.metrics.avg_latency for r in results[framework]) / len(
                    results[framework]
                )
                avg_throughput = sum(
                    r.metrics.requests_per_second for r in results[framework]
                ) / len(results[framework])
                print(f"\n{framework.upper()}:")
                print(f"  Average Latency: {avg_latency*1000:.2f} ms")
                print(f"  Average Throughput: {avg_throughput:.2f} req/s")
                print(f"  Runs: {len(results[framework])}")

        return results


async def run_benchmark_cli(
    config_path: str,
    vllm_path: str,
    elasticmm_path: str,
    frameworks: list[str] | None = None,
) -> None:
    """CLI entry point for running benchmarks."""
    config = load_config(config_path)
    runner = BenchmarkRunner(
        config=config, vllm_path=vllm_path, elasticmm_path=elasticmm_path
    )

    await runner.run_comparison(frameworks=frameworks)
