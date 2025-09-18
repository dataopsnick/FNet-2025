#!/bin/bash
set -e

echo "🚀 Setting up FNet SageMaker Training Environment"

# Check AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-west-2}

echo "Account ID: ${ACCOUNT_ID}"
echo "Region: ${REGION}"

# Deploy CloudFormation stack
echo ""
echo "📦 Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file template.yaml \
    --stack-name fnet-sagemaker-stack \
    --parameter-overrides \
        BucketNamePrefix=sagemaker-fnet-experiments \
        RoleName=SageMaker-FNet-ExecutionRole \
    --capabilities CAPABILITY_NAMED_IAM \
    --region ${REGION}

# Get stack outputs
echo ""
echo "📊 Stack outputs:"
aws cloudformation describe-stacks \
    --stack-name fnet-sagemaker-stack \
    --query 'Stacks[0].Outputs' \
    --output table \
    --region ${REGION}

# Build and push Docker image
echo ""
echo "🐳 Building and pushing Docker image..."
bash scripts/build_and_push.sh

echo ""
echo "✅ Setup complete! You can now run training with:"
echo "   python scripts/launch_sagemaker_job.py"