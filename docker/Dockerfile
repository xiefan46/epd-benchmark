# EPD Benchmark Docker Image
# Base: RunPod PyTorch with CUDA 12.4
# Added: Miniconda for environment management

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="fxie46"
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
    && ${CONDA_DIR}/bin/conda init bash \
    && ${CONDA_DIR}/bin/conda config --set auto_activate_base false \
    && ${CONDA_DIR}/bin/conda clean -afy

# Create conda environment for EPD benchmark
RUN ${CONDA_DIR}/bin/conda create -n epd python=3.11 -y \
    && ${CONDA_DIR}/bin/conda clean -afy

# Add conda activation to bashrc
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate epd" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
