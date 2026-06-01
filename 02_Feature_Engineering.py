import os
import pandas as pd
import joblib
import boto3
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==========================================
# 1. PIPELINE CONFIGURATION
# ==========================================
# Set this to True to push artifacts to AWS
UPLOAD_TO_S3 = False 
ARTIFACT_DIR = './model_artifacts'
PROCESSED_DATA_DIR = './processed_tensors'

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

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
    try:
        import glob
        parquet_files = glob.glob('./local_data_lake/*.parquet')
        if not parquet_files:
            raise FileNotFoundError("No local parquet files found. Run 01_Ingestion first.")
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        df = df.sort_values(['route_id', 'timestamp']).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"⚠️ Local Load Failed: {e}. Attempting S3 Pull...")
        # Fallback to S3 if local data is missing but AWS is active
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        bucket_uri = f"s3://frye-data-lake-{account_id}/raw_data/"
        return pd.read_parquet(bucket_uri)

def sync_to_s3(local_path, s3_prefix, file_name):
    """Pushes the generated ML artifacts directly to the unique AWS Data Lake."""
    try:
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        bucket = f"frye-data-lake-{account_id}"
        s3 = boto3.client('s3')

        s3_key = f"{s3_prefix}{file_name}"
        s3.upload_file(local_path, bucket, s3_key)
        print(f"☁️ Synced {file_name} -> s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"🚨 Failed to sync {file_name} to S3: {e}")

def execute_engineering():
    df = load_data()
    print(f"✅ Loaded {len(df)} total historical rows.")

    # Encode Categoricals
    cabin_map = {'ECONOMY': 1, 'PREMIUM_ECONOMY': 2, 'BUSINESS': 3, 'FIRST': 4}
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

    # Save Locally
    data_path = os.path.join(PROCESSED_DATA_DIR, 'frye_scaled_training_data.parquet')
    scaler_path = os.path.join(ARTIFACT_DIR, 'frye_feature_scaler.pkl')
    encoder_path = os.path.join(ARTIFACT_DIR, 'frye_label_encoders.pkl')

    scaled_df.to_parquet(data_path, engine='pyarrow', index=False)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoders, encoder_path)
    print("📦 Local ML Artifacts Secured.")

    # Push to AWS Cloud
    if UPLOAD_TO_S3:
        print("\n🚀 Initiating Cloud Sync...")
        sync_to_s3(data_path, "processed_tensors/", "frye_scaled_training_data.parquet")
        sync_to_s3(scaler_path, "model_artifacts/", "frye_feature_scaler.pkl")
        sync_to_s3(encoder_path, "model_artifacts/", "frye_label_encoders.pkl")

if __name__ == "__main__":
    execute_engineering()
