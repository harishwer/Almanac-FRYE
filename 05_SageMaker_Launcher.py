import os
import json
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from dotenv import load_dotenv

# ==========================================
# 1. AWS CLOUD & CREDENTIALS MANAGEMENT
# ==========================================
# Detects if running on AWS Lambda vs Local Mac
IS_AWS = os.environ.get('AWS_EXECUTION_ENV') is not None

def get_credentials():
    """Dynamically routes credential fetches based on execution environment."""
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

def get_unique_bucket_name():
    """Appends AWS Account ID to prevent global namespace collisions."""
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    return f"frye-data-lake-{account_id}"

# ==========================================
# 2. SAGEMAKER ORCHESTRATION & MODEL PROMOTION
# ==========================================
def launch_training_cluster():
    print("🚀 Initializing FRYE SageMaker Orchestrator...")
    
    try:
        bucket = get_unique_bucket_name()
        sagemaker_session = sagemaker.Session(default_bucket=bucket)
        
        # When running locally, sagemaker.get_execution_role() often fails.
        # We explicitly pull the Role ARN from our Secrets/Env File first.
        role = creds.get('SAGEMAKER_ROLE_ARN')
        if not role:
            print("⚠️ No SAGEMAKER_ROLE_ARN found in secrets. Attempting auto-resolve...")
            role = sagemaker.get_execution_role()
            
    except Exception as e:
        raise RuntimeError(f"AWS Auth Failure. Ensure roles and policies are configured: {e}")

    # Define the SageMaker Estimator (The Environment)
    huggingface_estimator = HuggingFace(
        entry_point='03_Distributed_Training.py', 
        source_dir='.',                           
        instance_type='ml.g4dn.xlarge',           # NVIDIA T4 GPU for high-speed tensor processing
        instance_count=1,
        role=role,
        transformers_version='4.28.1',            
        pytorch_version='2.0.0',                  
        py_version='py310',
        output_path=f"s3://{bucket}/model_output/", 
        hyperparameters={'epochs': 50}
    )

    print(f"🔥 Launching Training Cluster. Binding to s3://{bucket}/processed_tensors/ ...")
    
    # Executing the fit function. 
    # Lambda will 'wait' here until training finishes. (Ensure Lambda timeout is set to 15 mins).
    huggingface_estimator.fit({
        'training': f's3://{bucket}/processed_tensors/'
    }, wait=True)

    # ---------------------------------------------------------
    # MODEL PROMOTION (The CI/CD Link to the Live API)
    # ---------------------------------------------------------
    print("\n📦 Training Complete. Executing Model Promotion to 'Latest' Tag...")
    s3 = boto3.client('s3')

    # Get the exact randomized S3 URI SageMaker just generated
    source_uri = huggingface_estimator.model_data
    source_key = source_uri.replace(f"s3://{bucket}/", "")

    # Copy it to a static endpoint for the API to pull on its next reboot
    s3.copy_object(
        CopySource={'Bucket': bucket, 'Key': source_key},
        Bucket=bucket,
        Key='model_output/latest_model.tar.gz'
    )

    print(f"✅ Neural Network promoted successfully to: s3://{bucket}/model_output/latest_model.tar.gz")
    print("   The 04_Live_Endpoint API will pull this updated brain on its next boot.")

# ==========================================
# 3. AWS LAMBDA ENTRY POINT
# ==========================================
def lambda_handler(event, context):
    """
    Entry point for AWS EventBridge.
    CRITICAL: Go to Lambda Configuration -> General Configuration -> Set Timeout to 15 minutes.
    """
    print("☁️ AWS Lambda Trigger Detected: Firing SageMaker Retraining Pipeline...")
    launch_training_cluster()
    return {
        "statusCode": 200, 
        "body": "SageMaker Training & Promotion Executed Successfully"
    }

# ==========================================
# 4. LOCAL EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("💻 Local Terminal Execution Detected.")
    launch_training_cluster()

