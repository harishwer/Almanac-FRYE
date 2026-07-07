import os
import json
import boto3
import math
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from dotenv import load_dotenv

# ==========================================
# 1. AWS CLOUD & CREDENTIALS MANAGEMENT
# ==========================================
IS_AWS = os.environ.get('AWS_EXECUTION_ENV') is not None

def get_credentials():
    if IS_AWS:
        try:
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId='frye_api_keys')
            return json.loads(response['SecretString'])
        except Exception as e:
            print(f"🛑 AWS Secrets Manager Error: {e}")
            return {}
    else:
        load_dotenv()
        return os.environ

creds = get_credentials()

def get_aws_context():
    try:
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        bucket_name = f"frye-data-lake-{account_id}"
        sagemaker_role = f"arn:aws:iam::{account_id}:role/FRYE_SageMaker_Execution_Role"
        return bucket_name, sagemaker_role
    except Exception as e:
        raise RuntimeError(f"Failed to fetch AWS STS Caller Identity: {e}")

# ==========================================
# 2. AUTO-SCALING HARDWARE HEURISTICS
# ==========================================
def calculate_s3_size_gb(bucket, prefix="processed_tensors/"):
    """Calculates the exact total volume of the training dataset in S3."""
    print(f"📊 Analyzing S3 Data Lake Volume: s3://{bucket}/{prefix} ...")
    s3 = boto3.client('s3')
    total_bytes = 0
    paginator = s3.get_paginator('list_objects_v2')

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                total_bytes += obj['Size']
    except Exception as e:
        print(f"⚠️ S3 Access Error: {e}. Defaulting estimation to 1 GB.")
        return 1.0

    size_gb = total_bytes / (1024 ** 3)
    if size_gb == 0:
        raise ValueError("Data Lake is empty. Run ingestion first.")

    print(f"   ↳ Total Parquet Volume Detected: {size_gb:.4f} GB")
    return size_gb

def determine_optimal_instance(data_size_gb):
    """
    Selects the cheapest sufficient hardware.
    Checks CPU instances first for small data, then scales up to multi-GPU.
    """
    expansion_factor = 3.5  # Parquet compression -> Sequence Array Overhead

    # Pre-defined enterprise hardware tiers, ordered by cheapest total execution cost
    catalog = [
        {"type": "ml.m5.large",      "gpus": 0, "ram": 8,   "cost": 0.115, "nodes": 1},
        {"type": "ml.c5.xlarge",     "gpus": 0, "ram": 8,   "cost": 0.204, "nodes": 1},
        {"type": "ml.c5.2xlarge",    "gpus": 0, "ram": 16,  "cost": 0.408, "nodes": 1},
        {"type": "ml.g4dn.xlarge",   "gpus": 1, "ram": 16,  "cost": 0.736, "nodes": 1},
        {"type": "ml.g5.2xlarge",    "gpus": 1, "ram": 32,  "cost": 1.515, "nodes": 1},
        {"type": "ml.g5.12xlarge",   "gpus": 4, "ram": 192, "cost": 7.090, "nodes": 1}
    ]

    print("🧠 Executing Hardware Selection Heuristic...")

    for instance in catalog:
        # CPUs use 1 process. Distributed GPUs spawn 1 process per GPU, duplicating RAM needs.
        active_processes = max(1, instance["gpus"])
        estimated_ram_needed = data_size_gb * expansion_factor * active_processes

        # Ensure we don't exceed 85% of total system memory (leave room for OS/CUDA drivers)
        if instance["ram"] * 0.85 > estimated_ram_needed:
            print(f"   ↳ Selected {instance['type']} (x{instance['nodes']})")
            print(f"   ↳ Est. Memory Footprint: {estimated_ram_needed:.1f} GB / {instance['ram']} GB Available")
            return instance['type'], instance['nodes']

    raise MemoryError(
        f"Data volume ({data_size_gb:.1f} GB) exceeds maximum cloud node memory "
        f"for a fully-replicated loading strategy. Data layer optimization required."
    )

# ==========================================
# 3. SAGEMAKER ORCHESTRATION
# ==========================================
def launch_training_cluster():
    print("🚀 Initializing FRYE SageMaker Orchestrator...")

    bucket, role_arn = get_aws_context()
    role = creds.get('SAGEMAKER_ROLE_ARN', role_arn)

    # 1. Dynamically scale cluster based on current Data Lake size
    data_size_gb = calculate_s3_size_gb(bucket)
    instance_type, instance_count = determine_optimal_instance(data_size_gb)

    # 2. Auto-generate requirements.txt to inject missing libraries into the PyTorch container
    print("📝 Generating requirements.txt for the SageMaker container...")
    with open("requirements.txt", "w") as f:
        f.write("transformers==4.28.1\naccelerate\npandas\npyarrow\n")

    # 3. Define the PyTorch Estimator (Guarantees CPU/GPU compatibility)
    pytorch_estimator = PyTorch(
        entry_point='03_Distributed_Training.py',
        source_dir='.',
        instance_type=instance_type,
        instance_count=instance_count,
        role=role,
        framework_version='2.0.0',
        py_version='py310',
        base_job_name='frye-pricing-engine',  # Custom prefix for billing and tracking
        output_path=f"s3://{bucket}/model_output/",
        hyperparameters={'epochs': 50}
    )

    print(f"🔥 Launching Cloud Cluster. Binding to s3://{bucket}/processed_tensors/ ...")

    # Enable FastFile (Lazy Streaming) and ShardedByS3Key (Distributed Data Parallel Sharding)
    training_data_config = TrainingInput(
        s3_data=f's3://{bucket}/processed_tensors/',
        distribution='ShardedByS3Key',
        input_mode='FastFile'
    )

    pytorch_estimator.fit({'training': training_data_config}, wait=True)

    # ---------------------------------------------------------
    # MODEL PROMOTION (CI/CD)
    # ---------------------------------------------------------
    print("\n📦 Training Complete. Executing Model Promotion to 'Latest' Tag...")
    s3 = boto3.client('s3')

    source_uri = pytorch_estimator.model_data
    source_key = source_uri.replace(f"s3://{bucket}/", "")

    s3.copy_object(
        CopySource={'Bucket': bucket, 'Key': source_key},
        Bucket=bucket,
        Key='model_output/latest_model.tar.gz'
    )

    print(f"✅ Neural Network promoted successfully to: s3://{bucket}/model_output/latest_model.tar.gz")

# ==========================================
# 4. ENTRY POINTS
# ==========================================
def lambda_handler(event, context):
    print("☁️ AWS Lambda Trigger Detected: Firing SageMaker Pipeline...")
    launch_training_cluster()
    return {"statusCode": 200, "body": "Success"}

if __name__ == "__main__":
    print("💻 Local Terminal Execution Detected.")
    launch_training_cluster()
