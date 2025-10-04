#!/bin/bash
set -e

# Configuration
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-west-2}
REPO_NAME="causal-fnet-repo"
IMAGE_TAG="latest"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "🚀 Building and pushing FNet Docker image to ECR"
echo "Account: ${ACCOUNT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPO_NAME}"

# Login to AWS's ECR Public Registry for SageMaker base images
echo "📦 Logging into AWS ECR for base images..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com

# Login to your ECR registry
echo "📦 Logging into your ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Create repository if it doesn't exist
if ! aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} > /dev/null 2>&1; then
    echo "📁 Creating ECR repository: ${REPO_NAME}"
    aws ecr create-repository \
        --repository-name ${REPO_NAME} \
        --region ${REGION} \
        --image-scanning-configuration scanOnPush=true
else
    echo "✅ Repository ${REPO_NAME} already exists"
fi

# Build the Docker image
#echo "🔨 Building Docker image..."
#docker build -t ${IMAGE_URI} --build-arg REGION=${REGION} .

echo "🚀 Building and pushing multi-platform image for linux/amd64..."
docker buildx build \
  --platform linux/amd64 \
  --build-arg REGION=${REGION} \
  -t ${IMAGE_URI} \
  --push .

# Run a quick test to ensure the image works
echo "🧪 Testing Docker image..."
docker run --rm ${IMAGE_URI} python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Push to ECR
echo "⬆️ Pushing image to ECR..."
docker push ${IMAGE_URI}

echo "✅ Successfully pushed image: ${IMAGE_URI}"
echo ""
echo "Next steps:"
echo "1. Deploy CloudFormation stack: aws cloudformation deploy --template-file template.yaml --stack-name fnet-sagemaker-stack --capabilities CAPABILITY_NAMED_IAM"
echo "2. Run training: python scripts/launch_sagemaker_job.py"
