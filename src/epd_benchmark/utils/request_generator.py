"""
Dynamic Request Generator for EPD Benchmarks.

Generates mixed text-only and multimodal requests with configurable rates
and variance to simulate realistic workloads.
"""

import asyncio
import base64
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import aiohttp
from PIL import Image


@dataclass
class RequestStats:
    """Statistics for request generation."""

    total_requests_sent: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    text_requests: int = 0
    multimodal_requests: int = 0

    # Latency tracking
    latencies: list[float] = field(default_factory=list)
    ttft_list: list[float] = field(default_factory=list)  # Time to first token

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        return sorted_latencies[len(sorted_latencies) // 2]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class DynamicRequestGenerator:
    """
    Generates dynamic workloads with varying request rates.

    Supports both text-only and multimodal requests with configurable
    base rates and variance.
    """

    def __init__(
        self,
        proxy_url: str,
        model_name: str,
        image_dir: str | None = None,
        text_data_path: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.proxy_url = proxy_url
        self.model_name = model_name
        self.image_dir = Path(image_dir) if image_dir else None
        self.text_data_path = Path(text_data_path) if text_data_path else None
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.stats = RequestStats()
        self._running = False
        self._session: aiohttp.ClientSession | None = None

        # Load data
        self._images: list[Path] = []
        self._text_prompts: list[str] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load images and text prompts from configured paths."""
        # Load images
        if self.image_dir and self.image_dir.exists():
            self._images = list(self.image_dir.glob("*.jpg")) + list(
                self.image_dir.glob("*.png")
            )
            print(f"Loaded {len(self._images)} images from {self.image_dir}")

        # Load text prompts
        if self.text_data_path and self.text_data_path.exists():
            with open(self.text_data_path) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "conversations" in data:
                            for conv in data["conversations"]:
                                if conv.get("from") == "human":
                                    self._text_prompts.append(conv.get("value", ""))
                        elif "prompt" in data:
                            self._text_prompts.append(data["prompt"])
                        elif "text" in data:
                            self._text_prompts.append(data["text"])
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {len(self._text_prompts)} text prompts from {self.text_data_path}")

        # Default prompts if none loaded
        if not self._text_prompts:
            self._text_prompts = [
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a short poem about the ocean.",
                "What are the benefits of exercise?",
                "Describe the process of photosynthesis.",
            ]

    def _encode_image_base64(self, image_path: Path) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_image_url(self, image_path: Path) -> str:
        """Get image URL (base64 data URL or file URL)."""
        # Use base64 encoding for broader compatibility
        ext = image_path.suffix.lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        base64_data = self._encode_image_base64(image_path)
        return f"data:{mime_type};base64,{base64_data}"

    def _create_text_request(self) -> dict[str, Any]:
        """Create a text-only request."""
        prompt = random.choice(self._text_prompts)
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

    def _create_multimodal_request(self) -> dict[str, Any]:
        """Create a multimodal request with image."""
        if not self._images:
            # Fall back to text request if no images
            return self._create_text_request()

        image_path = random.choice(self._images)
        image_url = self._get_image_url(image_path)

        prompts = [
            "What is in this image?",
            "Describe this image in detail.",
            "What objects can you see in this image?",
            "What is happening in this image?",
            "Analyze this image and describe its contents.",
        ]

        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": random.choice(prompts)},
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

    async def _send_request(self, request_data: dict[str, Any], is_multimodal: bool) -> None:
        """Send a single request and track metrics."""
        request_id = str(uuid.uuid4())
        headers = {"Content-Type": "application/json", "x-request-id": request_id}

        start_time = time.perf_counter()
        self.stats.total_requests_sent += 1

        if is_multimodal:
            self.stats.multimodal_requests += 1
        else:
            self.stats.text_requests += 1

        try:
            async with self._session.post(
                self.proxy_url, json=request_data, headers=headers
            ) as response:
                if response.status == 200:
                    await response.json()  # Consume the response
                    latency = time.perf_counter() - start_time
                    self.stats.successful_requests += 1
                    self.stats.latencies.append(latency)
                else:
                    self.stats.failed_requests += 1
                    error_text = await response.text()
                    print(f"Request failed with status {response.status}: {error_text[:200]}")
        except Exception as e:
            self.stats.failed_requests += 1
            print(f"Request error: {e}")

    async def _generate_requests(
        self,
        duration_seconds: float,
        text_base_rate: float,
        multimodal_base_rate: float,
        text_variance: float,
        multimodal_variance: float,
    ) -> None:
        """Generate requests at specified rates with variance."""
        start_time = time.time()
        end_time = start_time + duration_seconds

        # Track next request times
        next_text_time = start_time
        next_mm_time = start_time

        while self._running and time.time() < end_time:
            current_time = time.time()
            tasks = []

            # Generate text requests
            if current_time >= next_text_time and text_base_rate > 0:
                request_data = self._create_text_request()
                tasks.append(self._send_request(request_data, is_multimodal=False))

                # Calculate next text request time with variance
                base_interval = 1.0 / text_base_rate
                variance = random.uniform(-text_variance, text_variance) * base_interval
                next_text_time = current_time + base_interval + variance

            # Generate multimodal requests
            if current_time >= next_mm_time and multimodal_base_rate > 0:
                request_data = self._create_multimodal_request()
                tasks.append(self._send_request(request_data, is_multimodal=True))

                # Calculate next multimodal request time with variance
                base_interval = 1.0 / multimodal_base_rate
                variance = random.uniform(-multimodal_variance, multimodal_variance) * base_interval
                next_mm_time = current_time + base_interval + variance

            # Execute pending requests
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Small sleep to prevent busy waiting
            await asyncio.sleep(0.01)

    async def run(
        self,
        duration_seconds: float = 600,
        text_base_rate: float = 30.0,
        multimodal_base_rate: float = 20.0,
        text_variance: float = 0.3,
        multimodal_variance: float = 1.5,
        on_progress: Callable[[RequestStats], None] | None = None,
    ) -> RequestStats:
        """
        Run the request generator for the specified duration.

        Args:
            duration_seconds: How long to run the benchmark
            text_base_rate: Base rate for text requests (requests/second)
            multimodal_base_rate: Base rate for multimodal requests (requests/second)
            text_variance: Variance factor for text request rate
            multimodal_variance: Variance factor for multimodal request rate
            on_progress: Optional callback for progress updates

        Returns:
            RequestStats with final statistics
        """
        self._running = True
        self.stats = RequestStats()

        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=100)
        self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

        try:
            print(f"Starting workload generation for {duration_seconds}s")
            print(f"  Text rate: {text_base_rate} req/s (variance: {text_variance})")
            print(f"  Multimodal rate: {multimodal_base_rate} req/s (variance: {multimodal_variance})")

            # Start progress reporter
            progress_task = None
            if on_progress:

                async def report_progress():
                    while self._running:
                        on_progress(self.stats)
                        await asyncio.sleep(5)

                progress_task = asyncio.create_task(report_progress())

            # Generate requests
            await self._generate_requests(
                duration_seconds,
                text_base_rate,
                multimodal_base_rate,
                text_variance,
                multimodal_variance,
            )

            if progress_task:
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

        finally:
            self._running = False
            if self._session:
                await self._session.close()

        return self.stats

    def stop(self) -> None:
        """Stop the request generator."""
        self._running = False

    @property
    def total_requests_sent(self) -> int:
        return self.stats.total_requests_sent

    @property
    def successful_requests(self) -> int:
        return self.stats.successful_requests
