import os
import platform
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
from accelerate import Accelerator

# AWS SageMaker Cloud Paths
SM_TRAINING_DIR = os.environ.get('SM_CHANNEL_TRAINING')
PROCESSED_DATA_DIR = SM_TRAINING_DIR if SM_TRAINING_DIR else './processed_tensors'
MODEL_DIR = './model_artifacts'
os.makedirs(MODEL_DIR, exist_ok=True)

CONTEXT_LENGTH = 7
MAX_LAG = 1
HISTORY_LENGTH = CONTEXT_LENGTH + MAX_LAG

FEATURES = [
    'route_id', 'service_class_tier', 'bucket_tier', 'booking_window_days',
    'is_direct', 'route_popularity', 'carrier_competition_count', 'airport_hub_type',
    'day_of_week', 'seasonality_index', 'time_of_flight_block', 'fuel_price_index',
    'exchange_rate_delta', 'airport_tax_fee', 'ancillary_score'
]
TARGET = 'competitor_price'

def probe_hardware_environment():
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

class FRYEContinuousPricingEngine(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        config = TimeSeriesTransformerConfig(
            prediction_length=1, context_length=CONTEXT_LENGTH, input_size=num_features,
            encoder_layers=2, decoder_layers=2, d_model=64, num_time_features=1, lags_sequence=[MAX_LAG]
        )
        self.transformer = TimeSeriesTransformerModel(config)
        self.price_prediction_head = nn.Linear(64, 1)
        self.activation = nn.Sigmoid() # Squashes output to positive bounds

    def forward(self, past_values):
        mask = torch.ones_like(past_values)
        time_features = torch.zeros((past_values.shape[0], past_values.shape[1], 1), device=past_values.device)
        outputs = self.transformer(past_values=past_values, past_observed_mask=mask, past_time_features=time_features)
        raw_output = self.price_prediction_head(outputs.encoder_last_hidden_state[:, -1, :])
        return self.activation(raw_output)

def build_training_sequences():
    data_path = os.path.join(PROCESSED_DATA_DIR, 'frye_scaled_training_data.parquet')
    df = pd.read_parquet(data_path)
    X, Y = [], []

    # Requires REAL data history >= 8 rows per route
    for route_val in df['route_id'].unique():
        route_df = df[df['route_id'] == route_val].reset_index(drop=True)
        if len(route_df) < HISTORY_LENGTH + 1:
            continue # Skips route if insufficient real data exists

        feature_matrix = route_df[FEATURES].values
        target_array = route_df[TARGET].values

        for i in range(len(route_df) - HISTORY_LENGTH):
            X.append(feature_matrix[i : i + HISTORY_LENGTH])
            Y.append([target_array[i + HISTORY_LENGTH]])

    # --- NEW SAFEGUARD ---
    if len(X) == 0:
        raise RuntimeError(f"🚨 Not enough data! The model requires at least {HISTORY_LENGTH + 1} historical records for a single route to build one training sequence. Run your ingestion pipeline a few more times.")

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32)
    return DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=16, shuffle=True)


def train_engine():
    safe_device = probe_hardware_environment()
    if safe_device == "cpu": os.environ["ACCELERATE_USE_CPU"] = "true"

    accelerator = Accelerator()
    model = FRYEContinuousPricingEngine(num_features=len(FEATURES))
    optimizer = AdamW(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    train_dataloader = build_training_sequences()
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    model.train()
    for epoch in range(50):
        for batch_x, batch_y in train_dataloader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            accelerator.backward(loss)
            optimizer.step()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        sagemaker_dir = os.environ.get('SM_MODEL_DIR')
        save_path = os.path.join(sagemaker_dir if sagemaker_dir else MODEL_DIR, "frye_continuous_pricing_weights.pt")
        torch.save(unwrapped.state_dict(), save_path)
        print(f"✅ Training Complete. Secured at: {save_path}")

if __name__ == "__main__": train_engine()
