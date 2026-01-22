"""
ElasticMM Launcher

Manages the lifecycle of ElasticMM system for benchmarking.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

from epd_benchmark.config import BenchmarkConfig


class ElasticMMLauncher:
    """
    Manages ElasticMM system lifecycle for benchmarking.

    Uses the ElasticMM system module to start/stop the complete system.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        elasticmm_path: str | Path,
    ):
        self.config = config
        self.elasticmm_path = Path(elasticmm_path)
        self._system = None
        self._proxy_url: str | None = None

        # Add ElasticMM to Python path
        sys.path.insert(0, str(self.elasticmm_path))

    async def start(self) -> str:
        """
        Start the ElasticMM system.

        Returns:
            The proxy URL for sending requests
        """
        print("Starting ElasticMM system...")

        # Import ElasticMM modules (requires elasticmm to be installed or in path)
        try:
            from elasticmm.system import ElasticMMSystem, SystemConfig
        except ImportError as e:
            raise ImportError(
                f"Failed to import ElasticMM. Ensure it's installed: {e}\n"
                f"Run: pip install -e {self.elasticmm_path}"
            ) from e

        # Create system config
        total_gpus = self.config.gpus.elasticmm.text_gpus + self.config.gpus.elasticmm.multimodal_gpus

        system_config = SystemConfig(
            total_gpus=total_gpus,
            text_gpus=self.config.gpus.elasticmm.text_gpus,
            multimodal_gpus=self.config.gpus.elasticmm.multimodal_gpus,
            model_path=self.config.model.name,
            proxy_host="0.0.0.0",
            proxy_port=self.config.ports.elasticmm.proxy,
            sd_host="0.0.0.0",
            sd_port=self.config.ports.elasticmm.service_discovery,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )

        print(f"  Total GPUs: {total_gpus}")
        print(f"  Text GPUs: {self.config.gpus.elasticmm.text_gpus}")
        print(f"  Multimodal GPUs: {self.config.gpus.elasticmm.multimodal_gpus}")
        print(f"  Model: {self.config.model.name}")

        # Create and start system
        self._system = ElasticMMSystem(system_config)

        try:
            await self._system.start()
            print("  System started, waiting for proxy to stabilize...")

            # Wait for proxy to be ready
            await asyncio.sleep(5)

            proxy_port = self.config.ports.elasticmm.proxy
            if not await self._wait_for_server(proxy_port):
                raise RuntimeError(f"ElasticMM proxy not ready on port {proxy_port}")

            self._proxy_url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
            print(f"ElasticMM ready at {self._proxy_url}")
            return self._proxy_url

        except Exception as e:
            print(f"Failed to start ElasticMM: {e}")
            await self.stop()
            raise

    async def _wait_for_server(self, port: int, timeout: int = 300) -> bool:
        """Wait for a server to become ready."""
        url = f"http://localhost:{port}/health"
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                try:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            return True
                except Exception:
                    pass
                await asyncio.sleep(1)

        # Also try the chat completions endpoint as a fallback
        url = f"http://localhost:{port}/v1/chat/completions"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    # Send a minimal request to check if the server is up
                    async with session.options(url, timeout=5) as response:
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)

        return False

    async def stop(self) -> None:
        """Stop the ElasticMM system."""
        print("Stopping ElasticMM system...")

        if self._system:
            try:
                await asyncio.wait_for(self._system.stop(), timeout=30)
                print("ElasticMM system stopped gracefully")
            except asyncio.TimeoutError:
                print("ElasticMM stop timed out, forcing shutdown...")
                # Try to shutdown Ray as a last resort
                try:
                    import ray

                    if ray.is_initialized():
                        ray.shutdown()
                except Exception as e:
                    print(f"Ray shutdown warning: {e}")
            except Exception as e:
                print(f"Error stopping ElasticMM: {e}")

        self._system = None
        self._proxy_url = None

    @property
    def proxy_url(self) -> str | None:
        """Get the proxy URL."""
        return self._proxy_url

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False


class ElasticMMLauncherSync:
    """
    Synchronous wrapper for ElasticMM launcher.

    Useful for scripts that don't use asyncio.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        elasticmm_path: str | Path,
    ):
        self._launcher = ElasticMMLauncher(config, elasticmm_path)
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> str:
        """Start ElasticMM system synchronously."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(self._launcher.start())

    def stop(self) -> None:
        """Stop ElasticMM system synchronously."""
        if self._loop:
            self._loop.run_until_complete(self._launcher.stop())
            self._loop.close()
            self._loop = None

    @property
    def proxy_url(self) -> str | None:
        return self._launcher.proxy_url

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
