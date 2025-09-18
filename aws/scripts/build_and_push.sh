#!/bin/bash

# --- Configuration ---
# Your AWS account ID and region.
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-east-1" # e.g., us-east-1, us-west-2

# The name for your ECR repository and image tag.
REPO_NAME="causal-fnet-repo"
IMAGE_TAG="latest"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

# --- Script Logic ---
# Login to Amazon ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Create the ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Creating ECR repository: ${REPO_NAME}"
    aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION} > /dev/null
else
    echo "Repository ${REPO_NAME} already exists."
fi

# Build the Docker image
echo "Building Docker image: ${IMAGE_URI}"
docker build -t ${IMAGE_URI} .

# Push the image to ECR
echo "Pushing image to ECR..."
docker push ${IMAGE_URI}

echo "Image push complete: ${IMAGE_URI}"