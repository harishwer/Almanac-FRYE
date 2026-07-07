import os
import platform
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
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

class StreamingRouteDataset(IterableDataset):
    """
    Lazy Streaming Layer. Streams data directly from S3 via SageMaker FastFile mount.
    Uses PyArrow to read record batches, maintaining near-zero RAM footprint.
    """
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    def __iter__(self):
        try:
            dataset = ds.dataset(self.data_dir, format="parquet")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize parquet dataset at {self.data_dir}: {e}")
            return

        route_buffers = {}
        # Stream from disk/S3 in tiny micro-batches
        for batch in dataset.to_batches():
            df = batch.to_pandas()
            for _, row in df.iterrows():
                route = row['route_id']
                if route not in route_buffers:
                    route_buffers[route] = []

                feat = [row[f] for f in FEATURES]
                targ = row[TARGET]
                route_buffers[route].append((feat, targ))

                # Yield sequence if buffer hits required length
                if len(route_buffers[route]) == HISTORY_LENGTH + 1:
                    X = [x[0] for x in route_buffers[route][:-1]]
                    Y = [route_buffers[route][-1][1]]
                    yield torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

                    # Pop oldest entry to advance the rolling window
                    route_buffers[route].pop(0)

def train_engine():
    safe_device = probe_hardware_environment()
    if safe_device == "cpu": os.environ["ACCELERATE_USE_CPU"] = "true"

    accelerator = Accelerator()
    model = FRYEContinuousPricingEngine(num_features=len(FEATURES))
    optimizer = AdamW(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    # Connect the Lazy Streamer (Notice shuffle=True is removed as IterableDatasets stream sequentially)
    dataset = StreamingRouteDataset(PROCESSED_DATA_DIR)
    train_dataloader = DataLoader(dataset, batch_size=16)

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

if __name__ == "__main__":
    train_engine()
