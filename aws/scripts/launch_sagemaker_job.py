# scripts/launch_sagemaker_job.py
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, CategoricalParameter
import boto3
import time

# --- Configuration ---
account_id = boto3.client("sts").get_caller_identity().get("Account")
region = "us-west-2"
image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/causal-fnet-repo:latest"
sagemaker_role = "arn:aws:iam::{account_id}:role/SageMakerExecutionRole" # UPDATE THIS
bucket = "your-sagemaker-bucket-name" # UPDATE THIS
s3_prefix = "fnet-p4d-grid-search"

# --- 1. Define the Distributed Training Estimator ---
# This configures SageMaker to use all 8 GPUs on the instance.
distribution = {
    "smdistributed": {
        "dataparallel": {
            "enabled": True
        }
    }
}

estimator = PyTorch( # Use the PyTorch estimator for easy distribution config
    entry_point="train.py",
    source_dir="../src",
    image_uri=image_uri,
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.p4d.24xlarge", # Using the A100-based instance
    output_path=f"s3://{bucket}/{s3_prefix}/output",
    sagemaker_session=sagemaker.Session(),
    distribution=distribution, # Apply the distributed training config
    hyperparameters={
        "num_train_epochs": 3, # Minimal epochs for cost efficiency
        "num_hidden_layers": 4,
        "intermediate_size": 4096, # Increased to match larger hidden sizes
        "learning_rate": 5e-4,
    },
    use_spot_instances=True,
    max_run=7200, # Allow 2 hours for each trial
    max_wait=8000,
)

# --- 2. Define the Ambitious 10x8 Grid for "Dataviz ROI" ---
hyperparameter_ranges = {
    "stft_window_size": CategoricalParameter([
        64, 128, 256, 384, 512, 768, 1024, 1536
    ]),
    "hidden_size": CategoricalParameter([
        256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560
    ]),
}

# --- 3. Define Objective Metric ---
objective_metric_name = "eval_loss"
metric_definitions = [{"Name": "eval_loss", "Regex": "'eval_loss': ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"}]

# --- 4. Create and Configure the HyperparameterTuner ---
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metric_definitions,
    strategy="Grid",
    objective_type="Minimize",
    max_jobs=80, # 10 hidden sizes * 8 window sizes = 80 jobs
    max_parallel_jobs=10, # Run up to 10 trials in parallel
)

# --- 5. Launch the Job ---
job_name = f"fnet-viz-p4d-grid-{int(time.time())}"
tuner.fit(job_name=job_name, wait=False) # Set wait=False to run in the background

print(f"--- Submitted Hyperparameter tuning job '{job_name}'. Monitor progress in the SageMaker console. ---")