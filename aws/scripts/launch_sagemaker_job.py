import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, CategoricalParameter, ContinuousParameter
import boto3
import time
from datetime import datetime

# ... (Configuration section remains the same) ...
account_id = boto3.client("sts").get_caller_identity().get("Account")
region = boto3.Session().region_name or "us-west-2"
role_name = "SageMaker-FNet-ExecutionRole"
bucket_prefix = "sagemaker-fnet-experiments"

sagemaker_role = f"arn:aws:iam::{account_id}:role/{role_name}"
bucket = f"{bucket_prefix}-{account_id}-{region}"
image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/causal-fnet-repo:latest"

print(f"Account ID: {account_id}")
print(f"Region: {region}")
print(f"Role ARN: {sagemaker_role}")
print(f"S3 Bucket: {bucket}")
print(f"Image URI: {image_uri}")

sagemaker_session = sagemaker.Session(default_bucket=bucket)

distribution = {
    "torch_distributed": {
        "enabled": True
    }
}

# Combine git info and training hyperparameters
# The entrypoint.sh script will parse --git_repo_url and --git_branch,
# and pass the rest to your train.py script.
hyperparameters={
    # === Git Configuration ===
    # IMPORTANT: Change these to your repository and desired branch
    "git_repo_url": "https://github.com/dataopsnick/FNet-2025.git",
    "git_branch": "sagemaker", # Example: a feature branch you are working on

    # === Original Training Hyperparameters ===
    "num_train_epochs": 3,
    "gradient_accumulation_steps": 4,
    "per_device_train_batch_size": 8,
    "learning_rate": 5e-4,
    "warmup_steps": 500,
    "weight_decay": 0.01,
}

# Create the PyTorch estimator
estimator = PyTorch(
    image_uri=image_uri,
    role=sagemaker_role,
    instance_count=1,
    instance_type= "ml.p3.8xlarge",
    volume_size=100,
    output_path=f"s3://{bucket}/fnet-training/output",
    code_location=f"s3://{bucket}/fnet-training/code",
    sagemaker_session=sagemaker_session,
    distribution=distribution,
    hyperparameters=hyperparameters, # Pass the combined hyperparameters
    metric_definitions=[
        {"Name": "train_loss", "Regex": r"'loss': (\S+)"},
        {"Name": "eval_loss", "Regex": r"'eval_loss': (\S+)"},
        {"Name": "eval_perplexity", "Regex": r"'eval_perplexity': (\S+)"},
    ],
    use_spot_instances=True,
    max_run=7200,
    max_wait=8000,
)

# Define hyperparameter ranges for tuning
hyperparameter_ranges = {
    # Model architecture parameters
    "hidden_size": CategoricalParameter([1024, 1536]), #[256, 384, 512, 768, 1024, 1280, 1536]),
    "num_hidden_layers": CategoricalParameter([4]), #[4, 6, 8, 10, 12]),
    "stft_window_size": CategoricalParameter([512, 1024]), #[64, 128, 256, 512, 1024]),
    
    # Training parameters  
    "learning_rate": ContinuousParameter(1e-5, 1e-3),
}

# Configure the tuner
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="eval_loss",
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=estimator.metric_definitions,
    strategy="Grid",
    objective_type="Minimize",
    max_jobs=4,
    max_parallel_jobs=2,
    early_stopping_type="Auto",
)

job_name = f"fnet-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

print(f"\nLaunching hyperparameter tuning job: {job_name}")
print("This will run in the background. Monitor progress in the SageMaker console.")

tuner.fit(
    job_name=job_name,
    wait=False
)

print(f"\n✅ Job submitted successfully!")
print(f"View progress at: https://console.aws.amazon.com/sagemaker/home?region={region}#/hyper-tuning-jobs/{job_name}")

print("\nTo retrieve the best model configuration later, use:")
print(f"tuner = HyperparameterTuner.attach('{job_name}')")
print("best_trial = tuner.best_training_job()")
print("print(best_trial)")