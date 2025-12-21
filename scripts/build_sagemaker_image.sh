#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <aws-account-id> <aws-region> <repository-name>"
  exit 1
fi

ACCOUNT_ID="$1"
REGION="$2"
REPO="$3"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest"

aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker build -f aws/Dockerfile.sagemaker -t "${REPO}:latest" .
docker tag "${REPO}:latest" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "Image pushed to ${IMAGE_URI}"
