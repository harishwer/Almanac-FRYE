import os
import pandas as pd
import joblib
import boto3
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==========================================
# 1. DYNAMIC CLOUD ENVIRONMENT DETECTION
# ==========================================
# Detects if running on AWS (Glue, Lambda, SageMaker, EC2) via common env vars
IS_AWS_ENV = any([
    os.environ.get('AWS_EXECUTION_ENV') is not None,
    os.environ.get('AWS_REGION') is not None,
    os.environ.get('AWS_DEFAULT_REGION') is not None, # <-- Catches AWS Glue
    os.environ.get('GLUE_COMMAND_CRITERIA') is not None # <-- Catches AWS Glue specific runtimes
])

UPLOAD_TO_S3 = True if IS_AWS_ENV else False # Forces true if running in AWS
# Dynamically construct S3 Bucket Name
try:
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    S3_BUCKET = f"frye-data-lake-{account_id}"
    S3_BASE_URI = f"s3://{S3_BUCKET}"
except Exception as e:
    S3_BUCKET = None
    if IS_AWS_ENV: raise RuntimeError(f"Failed to get AWS Account ID: {e}")

# Automatically route paths based on the compute environment
if IS_AWS_ENV:
    print(f"☁️ AWS Environment Detected. Routing paths to S3 Bucket: {S3_BUCKET}")
    RAW_DATA_LAKE = f"{S3_BASE_URI}/raw_data/"
    PROCESSED_DATA_DIR = f"{S3_BASE_URI}/processed_tensors/"
    MODEL_DIR = f"{S3_BASE_URI}/model_artifacts/"
else:
    print("💻 Local Environment Detected. Using local OS paths.")
    RAW_DATA_LAKE = './local_data_lake/'
    PROCESSED_DATA_DIR = './processed_tensors'
    MODEL_DIR = './model_artifacts'

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORICAL_FEATURES = ['route_id', 'service_class_tier', 'bucket_tier']
NUMERIC_FEATURES = [
    'competitor_price', 'booking_window_days', 'is_direct', 'route_popularity',
    'carrier_competition_count', 'airport_hub_type', 'day_of_week',
    'seasonality_index', 'time_of_flight_block', 'fuel_price_index',
    'exchange_rate_delta', 'airport_tax_fee', 'ancillary_score'
]

# ==========================================
# 2. DATA INGESTION & PROCESSING
# ==========================================
def load_data():
    if IS_AWS_ENV:
        try:
            print(f"📥 Loading raw data directly from {RAW_DATA_LAKE}...")
            return pd.read_parquet(RAW_DATA_LAKE)
        except Exception as e:
            raise RuntimeError(f"🚨 Failed to load data from S3 Data Lake: {e}")
    else:
        try:
            import glob
            parquet_files = glob.glob(f"{RAW_DATA_LAKE}*.parquet")
            if not parquet_files:
                raise FileNotFoundError("No local parquet files found. Run 01_Ingestion first.")
            df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
            df = df.sort_values(['route_id', 'timestamp']).reset_index(drop=True)
            return df
        except Exception as e:
            print(f"⚠️ Local Load Failed: {e}. Attempting Fallback S3 Pull...")
            if S3_BUCKET: return pd.read_parquet(f"{S3_BASE_URI}/raw_data/")
            raise e

def sync_to_s3(local_path, s3_prefix, file_name):
    """Pushes local artifacts to S3 (Used only when running locally)."""
    if not S3_BUCKET: return
    try:
        s3 = boto3.client('s3')
        s3_key = f"{s3_prefix}{file_name}"
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"☁️ Synced {file_name} -> s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"🚨 Failed to sync {file_name} to S3: {e}")

def execute_engineering():
    df = load_data()
    print(f"✅ Loaded {len(df)} total historical rows.")

    # Encode Categoricals
    cabin_map = {'ECONOMY': 1.0, 'PREMIUM_ECONOMY': 2.0, 'BUSINESS': 3.0, 'FIRST': 4.0}
    df['service_class_tier'] = df['service_class_tier'].map(cabin_map).fillna(1.0)

    encoders = {}
    for col in ['route_id', 'bucket_tier']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    scaled_matrix = scaler.fit_transform(df[all_features])
    scaled_df = pd.DataFrame(scaled_matrix, columns=all_features)
    scaled_df['timestamp'] = df['timestamp'].values

    # Determine Save Paths
    if IS_AWS_ENV:
        # Pandas and Joblib with s3fs can write directly to S3 URIs
        data_path = f"{PROCESSED_DATA_DIR}frye_scaled_training_data.parquet"
        scaler_path = f"{MODEL_DIR}frye_feature_scaler.pkl"
        encoder_path = f"{MODEL_DIR}frye_label_encoders.pkl"

        scaled_df.to_parquet(data_path, engine='pyarrow', index=False)

        # Joblib can write directly to S3 via s3fs, but for safety across all environments,
        # we write to ephemeral /tmp first, then upload via boto3
        import tempfile
        s3 = boto3.client('s3')

        with tempfile.NamedTemporaryFile() as tmp_scaler:
            joblib.dump(scaler, tmp_scaler.name)
            s3.upload_file(tmp_scaler.name, S3_BUCKET, "model_artifacts/frye_feature_scaler.pkl")

        with tempfile.NamedTemporaryFile() as tmp_encoder:
            joblib.dump(encoders, tmp_encoder.name)
            s3.upload_file(tmp_encoder.name, S3_BUCKET, "model_artifacts/frye_label_encoders.pkl")

        print(f"☁️ Cloud Processing Complete. Artifacts secured in {S3_BASE_URI}")

    else:
        # Save Locally & Sync
        data_path = os.path.join(PROCESSED_DATA_DIR, 'frye_scaled_training_data.parquet')
        scaler_path = os.path.join(MODEL_DIR, 'frye_feature_scaler.pkl')
        encoder_path = os.path.join(MODEL_DIR, 'frye_label_encoders.pkl')

        scaled_df.to_parquet(data_path, engine='pyarrow', index=False)
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoders, encoder_path)
        print("📦 Local ML Artifacts Secured.")

        if UPLOAD_TO_S3:
            print("\n🚀 Initiating Cloud Sync...")
            sync_to_s3(data_path, "processed_tensors/", "frye_scaled_training_data.parquet")
            sync_to_s3(scaler_path, "model_artifacts/", "frye_feature_scaler.pkl")
            sync_to_s3(encoder_path, "model_artifacts/", "frye_label_encoders.pkl")

if __name__ == "__main__":
    execute_engineering()
