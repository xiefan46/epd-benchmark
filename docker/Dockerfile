# EPD Benchmark Docker Image
# Base: RunPod PyTorch with CUDA 12.4
# Added: Miniconda for environment management

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="fxie46"
LABEL description="PyTorch + CUDA + Miniconda for EPD benchmarking"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git vim htop tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

RUN conda init bash \
    && conda config --set auto_activate_base false

WORKDIR /workspace
CMD ["/bin/bash"]
