# EPD Benchmark Docker Image
# Base: RunPod PyTorch with CUDA 12.4
# Added: Miniconda for environment management

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="xiefan46"
LABEL description="PyTorch + CUDA + Miniconda for EPD benchmarking"

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda init bash \
    && conda config --set auto_activate_base false \
    && conda clean -afy

# Create conda environment for EPD benchmark
# Using Python 3.11 to match the base image
RUN conda create -n epd python=3.11 -y \
    && conda clean -afy

# Install common dependencies in the epd environment
SHELL ["conda", "run", "-n", "epd", "/bin/bash", "-c"]

RUN pip install --no-cache-dir \
    aiohttp>=3.9.0 \
    httpx>=0.25.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    pyyaml>=6.0 \
    rich>=13.0.0 \
    click>=8.0.0 \
    Pillow>=10.0.0 \
    matplotlib>=3.7.0 \
    pytest>=7.0.0 \
    pytest-asyncio>=0.21.0

# Add conda activation to bashrc
RUN echo "conda activate epd" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
