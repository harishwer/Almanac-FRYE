import os
import torch
import joblib
import pandas as pd
import numpy as np
import boto3
import tarfile
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
import torch.nn as nn
from collections import deque

# ==========================================
# 1. DYNAMIC ENVIRONMENT & CACHE DETECTION
# ==========================================
# Detects if running on AWS (Lambda, ECS, AppRunner, EC2) via common env vars
IS_AWS_ENV = os.environ.get('AWS_EXECUTION_ENV') is not None or os.environ.get('AWS_REGION') is not None

# Allow manual override via terminal: export FORCE_S3_PULL=true
PULL_FROM_S3 = os.environ.get('FORCE_S3_PULL', str(IS_AWS_ENV)).lower() in ('true', '1', 't')

# Redis Configuration for AWS ElastiCache
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))

MODEL_DIR = '/tmp/model_artifacts' if PULL_FROM_S3 else './model_artifacts'
os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, 'frye_feature_scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'frye_label_encoders.pkl')
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'frye_continuous_pricing_weights.pt')

# Neural Architecture Constants (Must match training parameters precisely)
CONTEXT_LENGTH = 7
MAX_LAG = 1
HISTORY_LENGTH = CONTEXT_LENGTH + MAX_LAG
SCALER_FEATURES_COUNT = 16
MODEL_FEATURES_COUNT = 15
TARGET_INDEX = 3 # Index of 'competitor_price' in the scaled matrix

# Explicitly define the ordered feature array to prevent pandas schema drifting
SCALER_ORDERED_FEATURES = [
    'route_id', 'service_class_tier', 'bucket_tier', 'competitor_price',
    'booking_window_days', 'is_direct', 'route_popularity', 'carrier_competition_count',
    'airport_hub_type', 'day_of_week', 'seasonality_index', 'time_of_flight_block',
    'fuel_price_index', 'exchange_rate_delta', 'airport_tax_fee', 'ancillary_score'
]

# ==========================================
# 2. DISTRIBUTED FEATURE STORE (MEMORY BANK)
# ==========================================
# We define our cache clients globally so they persist between requests
local_memory_bank = {}
redis_client = None

if IS_AWS_ENV:
    import redis
    try:
        # decode_responses=True automatically decodes Redis byte strings to Python strings
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        redis_client.ping() # Validate connection on boot
        print(f"🔗 Connected to Distributed Cache: ElastiCache Redis at {REDIS_HOST}")
    except Exception as e:
        print(f"🚨 Redis Connection Failed. Ensure VPC/Security Groups allow port {REDIS_PORT}. Error: {e}")
        redis_client = None

def update_and_fetch_history(route: str, live_data: dict) -> list:
    """Seamlessly routes historical sequence retrieval to Redis or Local RAM."""
    if redis_client:
        # AWS ELASTICACHE ROUTING
        # 1. Push the new live JSON onto the right side of the Redis List
        redis_client.rpush(route, json.dumps(live_data))
        # 2. Trim the list to enforce our exact HISTORY_LENGTH (keep only the newest 8)
        redis_client.ltrim(route, -HISTORY_LENGTH, -1)

        # 3. Pull the updated sequence back from Redis
        raw_history = redis_client.lrange(route, 0, -1)
        history = [json.loads(x) for x in raw_history]

        # 4. Handle Cloud Cold Starts (If Redis has < 8 items, pad backward in time)
        while len(history) < HISTORY_LENGTH:
            history.insert(0, history[0])

        return history

    else:
        # LOCAL MAC FALLBACK ROUTING
        if route not in local_memory_bank:
            local_memory_bank[route] = deque([live_data] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        else:
            local_memory_bank[route].append(live_data)
        return list(local_memory_bank[route])

# ==========================================
# 3. NEURAL NETWORK CLASS (Must be redefined for loading)
# ==========================================
class FRYEContinuousPricingEngine(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        config = TimeSeriesTransformerConfig(
            prediction_length=1, context_length=CONTEXT_LENGTH, input_size=num_features,
            encoder_layers=2, decoder_layers=2, d_model=64, num_time_features=1, lags_sequence=[MAX_LAG]
        )
        self.transformer = TimeSeriesTransformerModel(config)
        self.price_prediction_head = nn.Linear(64, 1)
        self.activation = nn.Sigmoid()

    def forward(self, past_values):
        mask = torch.ones_like(past_values)
        time_features = torch.zeros((past_values.shape[0], past_values.shape[1], 1), device=past_values.device)
        outputs = self.transformer(past_values=past_values, past_observed_mask=mask, past_time_features=time_features)
        raw_output = self.price_prediction_head(outputs.encoder_last_hidden_state[:, -1, :])
        return self.activation(raw_output)

# ==========================================
# 4. AWS S3 BOOTSTRAPPER & LIFECYCLE
# ==========================================
def download_artifacts_from_s3():
    """Securely pulls the latest neural weights and scaling artifacts from S3 on boot."""
    try:
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        bucket_name = f"frye-data-lake-{account_id}"
        print(f"☁️ Cloud Mode Detected. Connecting to S3 Bucket: {bucket_name}...")
        s3 = boto3.client('s3')

        print("   📥 Downloading Feature Scalers and Encoders...")
        s3.download_file(bucket_name, "model_artifacts/frye_feature_scaler.pkl", SCALER_PATH)
        s3.download_file(bucket_name, "model_artifacts/frye_label_encoders.pkl", ENCODER_PATH)

        tar_path = os.path.join(MODEL_DIR, 'model.tar.gz')
        print("   📥 Downloading PyTorch Neural Weights (latest_model.tar.gz)...")
        s3.download_file(bucket_name, "model_output/latest_model.tar.gz", tar_path)

        print("   📦 Extracting neural architecture...")
        with tarfile.open(tar_path, "r:gz") as tar: tar.extractall(path=MODEL_DIR)
        print("✅ Cloud Bootstrap Complete.")
    except Exception as e:
        raise RuntimeError(f"🚨 Critical Failure pulling from S3: {e}")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Booting FRYE Inference Server...")

    if PULL_FROM_S3: download_artifacts_from_s3()
    else: print("💻 Local Mode Detected. Loading artifacts from local disk.")

    if not os.path.exists(SCALER_PATH) or not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(f"Missing ML Artifacts in {MODEL_DIR}. Ensure pipeline steps 02 & 03 completed.")

    ml_models["scaler"] = joblib.load(SCALER_PATH)
    ml_models["encoders"] = joblib.load(ENCODER_PATH)

    # We strictly map inference to the CPU. Since inference is batch-size 1,
    # CPU latency is negligible (~5ms) and avoids any Apple Silicon indexing crashes.
    device = torch.device('cpu')
    model = FRYEContinuousPricingEngine(num_features=MODEL_FEATURES_COUNT)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    model.to(device)

    ml_models["engine"] = model
    ml_models["device"] = device

    print("✅ Neural Engine loaded into RAM. API is now listening.")
    yield
    ml_models.clear()
    local_memory_bank.clear()

app = FastAPI(title="Project FRYE Continuous Pricing Engine", lifespan=lifespan)

# ==========================================
# 5. API DATA SCHEMAS (Pydantic Validation)
# ==========================================
class FlightContext(BaseModel):
    route_id: str; service_class_tier: str; bucket_tier: str; booking_window_days: float
    is_direct: float; route_popularity: float; carrier_competition_count: float
    airport_hub_type: float; day_of_week: float; seasonality_index: float
    time_of_flight_block: float; fuel_price_index: float; exchange_rate_delta: float
    airport_tax_fee: float; ancillary_score: float; competitor_price: float

class HybridInferenceRequest(BaseModel):
    # The client can send a single snapshot (API uses cache) OR a full history list (API bypasses cache)
    live_snapshot: FlightContext = None
    historical_context: list[FlightContext] = None

# ==========================================
# 6. THE INFERENCE ENDPOINT
# ==========================================
@app.post("/predict-price/")
async def predict_optimal_price(request: HybridInferenceRequest):

    # --- ROUTING LOGIC: STATELESS VS CACHED ---
    if request.historical_context:
        # STATELESS MODE: The client sent the full 8-day history. Bypass Redis entirely.
        if len(request.historical_context) != HISTORY_LENGTH:
            raise HTTPException(status_code=400, detail=f"Stateless mode requires exactly {HISTORY_LENGTH} snapshots.")

        historical_sequence = [item.dict() for item in request.historical_context]
        route = historical_sequence[-1]['route_id']
        baseline_price = historical_sequence[-1]['competitor_price']
        cache_status = "Bypassed (Client provided full history)"

    elif request.live_snapshot:
        # CACHED MODE: The client sent a single snapshot. Use Redis/Local Memory.
        live_data = request.live_snapshot.dict()
        route = live_data['route_id']
        baseline_price = live_data['competitor_price']

        historical_sequence = update_and_fetch_history(route, live_data)
        cache_status = "Warm Cache (Redis/RAM)" if len(set([str(x) for x in historical_sequence])) > 1 else "Cold Start (Padded Sequence)"

    else:
        raise HTTPException(status_code=400, detail="Must provide either 'live_snapshot' or 'historical_context'.")

    # --- CORE ML PIPELINE ---
    df = pd.DataFrame(historical_sequence)
    df = df[SCALER_ORDERED_FEATURES].copy()

    cabin_map = {'ECONOMY': 1.0, 'PREMIUM_ECONOMY': 2.0, 'BUSINESS': 3.0, 'FIRST': 4.0}
    df['service_class_tier'] = df['service_class_tier'].map(cabin_map).fillna(1.0)

    encoders = ml_models["encoders"]
    for col in ['route_id', 'bucket_tier']:
        try:
             # If the model sees a new route it wasn't trained on, safely fallback to the 0th index
            df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
            df[col] = encoders[col].transform(df[col])
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Encoding Error: {str(e)}")

    scaler = ml_models["scaler"]
    try:
        # Keep the data as a Pandas DataFrame so Scikit-Learn can read the feature names
        clean_df = df.astype(np.float32)
        scaled_matrix = scaler.transform(clean_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler Dimension Error: {str(e)}")

    model_matrix = np.delete(scaled_matrix, TARGET_INDEX, axis=1)
    tensor_input = torch.tensor(model_matrix, dtype=torch.float32).unsqueeze(0).to(ml_models["device"])

    with torch.no_grad(): # Disable gradient tracking for maximum speed
        scaled_prediction = ml_models["engine"](tensor_input)

    # 6. REVERSE SCALING
    # Create a dummy DataFrame with the correct column names to satisfy the inverse scaler
    dummy_df = pd.DataFrame(np.zeros((1, SCALER_FEATURES_COUNT)), columns=SCALER_ORDERED_FEATURES)
    dummy_df.iloc[0, TARGET_INDEX] = scaled_prediction.item()

    real_price = scaler.inverse_transform(dummy_df)[0, TARGET_INDEX]

    return {
        "status": "success",
        "route": route,
        "baseline_competitor_price": baseline_price,
        "frye_optimal_continuous_price": round(real_price, 2),
        "cache_status": cache_status
    }
