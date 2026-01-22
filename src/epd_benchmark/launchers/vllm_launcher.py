"""
vLLM EPD Launcher

Manages the lifecycle of vLLM disaggregated encoder-prefill-decode servers.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from epd_benchmark.config import BenchmarkConfig


@dataclass
class VLLMProcess:
    """Represents a vLLM server process."""

    role: str  # "encoder", "prefill", "decode", "proxy"
    process: subprocess.Popen
    port: int
    gpu: int | None = None
    log_file: Path | None = None


class VLLMLauncher:
    """
    Manages vLLM EPD server lifecycle.

    Starts encoder, prefill, decode servers and the EPD proxy.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        vllm_path: str | Path,
        log_dir: str | Path = "./logs",
    ):
        self.config = config
        self.vllm_path = Path(vllm_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.processes: list[VLLMProcess] = []
        self._ec_cache_path = Path(config.storage.ec_cache)

    def _get_ec_transfer_config(self, role: str) -> str:
        """Get EC transfer config JSON for the given role."""
        import json

        config = {
            "ec_connector": "ECExampleConnector",
            "ec_role": role,
            "ec_connector_extra_config": {"shared_storage_path": str(self._ec_cache_path)},
        }
        return json.dumps(config)

    def _get_kv_transfer_config(self, role: str) -> str:
        """Get KV transfer config JSON for the given role."""
        import json

        config = {"kv_connector": "NixlConnector", "kv_role": role}
        return json.dumps(config)

    def _start_encoder(self) -> VLLMProcess:
        """Start the encoder server."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"encoder_{timestamp}.log"

        gpu = self.config.gpus.vllm.encoder
        port = self.config.ports.vllm.encoder

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model.name,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            "0.01",
            "--enforce-eager",
            "--enable-request-id-headers",
            "--no-enable-prefix-caching",
            "--max-num-batched-tokens",
            "114688",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            self._get_ec_transfer_config("ec_producer"),
        ]

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(self.vllm_path)
            )

        return VLLMProcess(
            role="encoder", process=process, port=port, gpu=gpu, log_file=log_file
        )

    def _start_prefill(self) -> VLLMProcess:
        """Start the prefill server."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"prefill_{timestamp}.log"

        gpu = self.config.gpus.vllm.prefill
        port = self.config.ports.vllm.prefill

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["UCX_NET_DEVICES"] = "all"
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "5559"

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model.name,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            "0.7",
            "--enforce-eager",
            "--enable-request-id-headers",
            "--max-num-seqs",
            "128",
            "--ec-transfer-config",
            self._get_ec_transfer_config("ec_consumer"),
            "--kv-transfer-config",
            self._get_kv_transfer_config("kv_producer"),
        ]

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(self.vllm_path)
            )

        return VLLMProcess(
            role="prefill", process=process, port=port, gpu=gpu, log_file=log_file
        )

    def _start_decode(self) -> VLLMProcess:
        """Start the decode server."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"decode_{timestamp}.log"

        gpu = self.config.gpus.vllm.decode
        port = self.config.ports.vllm.decode

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["UCX_NET_DEVICES"] = "all"
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model.name,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            "0.7",
            "--enforce-eager",
            "--enable-request-id-headers",
            "--max-num-seqs",
            "128",
            "--kv-transfer-config",
            self._get_kv_transfer_config("kv_consumer"),
        ]

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(self.vllm_path)
            )

        return VLLMProcess(
            role="decode", process=process, port=port, gpu=gpu, log_file=log_file
        )

    def _start_proxy(self) -> VLLMProcess:
        """Start the EPD proxy server."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"proxy_{timestamp}.log"

        port = self.config.ports.vllm.proxy
        proxy_script = (
            self.vllm_path
            / "examples"
            / "online_serving"
            / "disaggregated_encoder"
            / "disagg_epd_proxy.py"
        )

        encode_url = f"http://localhost:{self.config.ports.vllm.encoder}"
        prefill_url = f"http://localhost:{self.config.ports.vllm.prefill}"
        decode_url = f"http://localhost:{self.config.ports.vllm.decode}"

        cmd = [
            sys.executable,
            str(proxy_script),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--encode-servers-urls",
            encode_url,
            "--prefill-servers-urls",
            prefill_url,
            "--decode-servers-urls",
            decode_url,
        ]

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(self.vllm_path)
            )

        return VLLMProcess(role="proxy", process=process, port=port, log_file=log_file)

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

        return False

    async def start(self) -> str:
        """
        Start all vLLM EPD servers.

        Returns:
            The proxy URL for sending requests
        """
        print("Starting vLLM EPD servers...")

        # Clear previous EC cache
        import shutil

        if self._ec_cache_path.exists():
            shutil.rmtree(self._ec_cache_path)
        self._ec_cache_path.mkdir(parents=True, exist_ok=True)

        # Start servers
        encoder_proc = self._start_encoder()
        self.processes.append(encoder_proc)
        print(f"  Started encoder on port {encoder_proc.port} (GPU {encoder_proc.gpu})")

        prefill_proc = self._start_prefill()
        self.processes.append(prefill_proc)
        print(f"  Started prefill on port {prefill_proc.port} (GPU {prefill_proc.gpu})")

        decode_proc = self._start_decode()
        self.processes.append(decode_proc)
        print(f"  Started decode on port {decode_proc.port} (GPU {decode_proc.gpu})")

        # Wait for servers to be ready
        print("  Waiting for servers to be ready...")
        for proc in [encoder_proc, prefill_proc, decode_proc]:
            if not await self._wait_for_server(proc.port):
                raise RuntimeError(f"Failed to start {proc.role} server on port {proc.port}")
            print(f"    {proc.role} ready")

        # Start proxy
        proxy_proc = self._start_proxy()
        self.processes.append(proxy_proc)

        if not await self._wait_for_server(proxy_proc.port):
            raise RuntimeError(f"Failed to start proxy on port {proxy_proc.port}")
        print(f"  Proxy ready on port {proxy_proc.port}")

        proxy_url = f"http://localhost:{proxy_proc.port}/v1/chat/completions"
        print(f"vLLM EPD ready at {proxy_url}")
        return proxy_url

    def stop(self) -> None:
        """Stop all vLLM servers."""
        print("Stopping vLLM EPD servers...")

        for proc in self.processes:
            if proc.process.poll() is None:
                print(f"  Stopping {proc.role}...")
                proc.process.terminate()
                try:
                    proc.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.process.kill()

        self.processes.clear()
        print("All vLLM servers stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
