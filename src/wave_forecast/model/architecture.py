import torch
import torch.nn as nn
import numpy as np
from ..utils.numerics import haversine_distance_vectorized

class SpatiotemporalTransformer(nn.Module):
    """The main transformer model for wave forecasting."""
    def __init__(self, config):
        super().__init__()
        model_cfg = config.model
        self.station_coords = config.stations.coordinates
        
        self.station_map = {sid: i for i, sid in enumerate(self.station_coords.keys())}
        self.station_embedding = nn.Embedding(len(self.station_coords), model_cfg.d_model)
        
        # Positional and feature embeddings
        self.feature_projection = nn.Linear(7, model_cfg.d_model) # Based on features from sequencer
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.training.seq_length, model_cfg.d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_cfg.d_model, nhead=model_cfg.num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_cfg.num_layers)
        
        # Output Head
        self.output_head = nn.Linear(model_cfg.d_model, 3) # Predicting Hs, Tp, MWD
        
    def forward(self, x, station_idx):
        # x shape: (batch, seq_len, features)
        
        # Add station and positional embeddings
        station_emb = self.station_embedding(station_idx).unsqueeze(1) # (batch, 1, d_model)
        x = self.feature_projection(x) + self.positional_encoding + station_emb
        
        # Pass through transformer
        output = self.transformer_encoder(x)
        
        # Final prediction
        prediction = self.output_head(output)
        
        return prediction