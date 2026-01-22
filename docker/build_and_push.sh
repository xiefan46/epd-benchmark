#!/bin/bash
# Build and push Docker image to Docker Hub
#
# Prerequisites:
#   1. Docker installed and running
#   2. Logged in to Docker Hub: docker login
#
# Usage:
#   ./docker/build_and_push.sh [tag]
#   ./docker/build_and_push.sh latest
#   ./docker/build_and_push.sh v1.0.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:-latest}"

# Docker Hub username
DOCKER_USER="${DOCKER_USER:-fxie46}"
IMAGE_NAME="epd-benchmark"
FULL_IMAGE_NAME="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"

echo "=================================================="
echo "Building Docker Image"
echo "=================================================="
echo "Image: $FULL_IMAGE_NAME"
echo "Context: $SCRIPT_DIR"
echo ""

# Build the image
echo "Building image..."
docker build -t "$FULL_IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

# Also tag as latest if not already latest
if [ "$TAG" != "latest" ]; then
    docker tag "$FULL_IMAGE_NAME" "${DOCKER_USER}/${IMAGE_NAME}:latest"
fi

echo ""
echo "Build complete!"
echo ""

# Ask before pushing
read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing to Docker Hub..."
    docker push "$FULL_IMAGE_NAME"
    if [ "$TAG" != "latest" ]; then
        docker push "${DOCKER_USER}/${IMAGE_NAME}:latest"
    fi
    echo ""
    echo "=================================================="
    echo "Push Complete!"
    echo "=================================================="
    echo ""
    echo "Image available at:"
    echo "  docker pull $FULL_IMAGE_NAME"
    echo ""
    echo "Use on RunPod:"
    echo "  Image: ${DOCKER_USER}/${IMAGE_NAME}:${TAG}"
    echo ""
else
    echo "Skipping push. To push later:"
    echo "  docker push $FULL_IMAGE_NAME"
fi
