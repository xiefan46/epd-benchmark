"""
Command-line interface for EPD benchmarks.
"""

import asyncio
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """EPD Benchmark Suite - Compare vLLM EPD vs ElasticMM performance."""
    pass


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="configs/benchmark_config.yaml",
    help="Path to benchmark configuration file",
)
@click.option(
    "--vllm-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to vLLM repository",
)
@click.option(
    "--elasticmm-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to ElasticMM repository",
)
@click.option(
    "--framework",
    "-f",
    type=click.Choice(["vllm", "elasticmm", "both"]),
    default="both",
    help="Which framework(s) to benchmark",
)
def run(config: str, vllm_path: str, elasticmm_path: str, framework: str):
    """Run the benchmark comparison."""
    from epd_benchmark.benchmarks.runner import run_benchmark_cli

    frameworks = None if framework == "both" else [framework]

    asyncio.run(
        run_benchmark_cli(
            config_path=config,
            vllm_path=vllm_path,
            elasticmm_path=elasticmm_path,
            frameworks=frameworks,
        )
    )


@main.command()
@click.argument("vllm_results", type=click.Path(exists=True))
@click.argument("elasticmm_results", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for comparison JSON",
)
def compare(vllm_results: str, elasticmm_results: str, output: str | None):
    """Compare results from two benchmark runs."""
    from epd_benchmark.utils.metrics import compare_results, print_comparison_report

    comparison = compare_results(vllm_results, elasticmm_results, output)
    print_comparison_report(comparison)


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="configs/benchmark_config.yaml",
    help="Path to benchmark configuration file",
)
@click.option(
    "--vllm-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to vLLM repository",
)
def start_vllm(config: str, vllm_path: str):
    """Start vLLM EPD servers (for manual testing)."""
    from epd_benchmark.config import load_config
    from epd_benchmark.launchers.vllm_launcher import VLLMLauncher

    cfg = load_config(config)
    launcher = VLLMLauncher(config=cfg, vllm_path=vllm_path)

    async def _start():
        proxy_url = await launcher.start()
        print(f"\nvLLM EPD running at: {proxy_url}")
        print("Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            launcher.stop()

    asyncio.run(_start())


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="configs/benchmark_config.yaml",
    help="Path to benchmark configuration file",
)
@click.option(
    "--elasticmm-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to ElasticMM repository",
)
def start_elasticmm(config: str, elasticmm_path: str):
    """Start ElasticMM system (for manual testing)."""
    from epd_benchmark.config import load_config
    from epd_benchmark.launchers.elasticmm_launcher import ElasticMMLauncher

    cfg = load_config(config)
    launcher = ElasticMMLauncher(config=cfg, elasticmm_path=elasticmm_path)

    async def _start():
        proxy_url = await launcher.start()
        print(f"\nElasticMM running at: {proxy_url}")
        print("Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await launcher.stop()

    asyncio.run(_start())


@main.command()
@click.option(
    "--url",
    "-u",
    type=str,
    required=True,
    help="Proxy URL to send requests to",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="Qwen/Qwen2.5-VL-3B-Instruct",
    help="Model name",
)
@click.option(
    "--duration",
    "-d",
    type=int,
    default=60,
    help="Duration in seconds",
)
@click.option(
    "--text-rate",
    type=float,
    default=10.0,
    help="Text requests per second",
)
@click.option(
    "--mm-rate",
    type=float,
    default=5.0,
    help="Multimodal requests per second",
)
@click.option(
    "--image-dir",
    type=click.Path(exists=True),
    help="Directory containing images for multimodal requests",
)
def workload(
    url: str,
    model: str,
    duration: int,
    text_rate: float,
    mm_rate: float,
    image_dir: str | None,
):
    """Generate workload against a running server."""
    from epd_benchmark.utils.request_generator import DynamicRequestGenerator

    async def _run():
        generator = DynamicRequestGenerator(
            proxy_url=url,
            model_name=model,
            image_dir=image_dir,
        )

        def on_progress(stats):
            print(
                f"Sent: {stats.total_requests_sent}, "
                f"Success: {stats.successful_requests}, "
                f"Avg Latency: {stats.avg_latency*1000:.1f}ms"
            )

        stats = await generator.run(
            duration_seconds=duration,
            text_base_rate=text_rate,
            multimodal_base_rate=mm_rate,
            on_progress=on_progress,
        )

        print("\n" + "=" * 40)
        print("Final Statistics:")
        print(f"  Total Requests: {stats.total_requests_sent}")
        print(f"  Successful: {stats.successful_requests}")
        print(f"  Failed: {stats.failed_requests}")
        print(f"  Text Requests: {stats.text_requests}")
        print(f"  Multimodal Requests: {stats.multimodal_requests}")
        print(f"  Avg Latency: {stats.avg_latency*1000:.2f}ms")
        print(f"  P50 Latency: {stats.p50_latency*1000:.2f}ms")
        print(f"  P99 Latency: {stats.p99_latency*1000:.2f}ms")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
