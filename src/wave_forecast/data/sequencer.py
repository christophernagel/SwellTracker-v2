import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class WaveSequenceDataset(Dataset):
    """PyTorch Dataset for creating wave forecast sequences."""
    def __init__(self, features_df, config, split='train'):
        self.config = config.training
        self.station_map = {sid: i for i, sid in enumerate(config.stations.active)}
        
        # Temporal split
        split_date = pd.to_datetime(self.config.train_split if split == 'train' else self.config.val_split)
        if split == 'train':
            self.data = features_df[features_df['timestamp'] < split_date].copy()
        else: # 'val' or 'test'
            self.data = features_df[features_df['timestamp'] >= split_date].copy()
            
        self.data['station_idx'] = self.data['station_id'].map(self.station_map)
        
        # Sequence generation
        self.sequences = self._generate_sequences()
        
    def _generate_sequences(self):
        sequences = []
        seq_len, pred_hor = self.config.seq_length, self.config.pred_horizon
        grouped = self.data.groupby('station_id')
        for station_id, group in grouped:
            num_samples = len(group)
            for i in range(num_samples - seq_len - pred_hor + 1):
                sequences.append({'station_id': station_id, 'start_idx': group.index[i]})
        return sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        station_id = seq_info['station_id']
        start_idx = seq_info['start_idx']
        
        station_data = self.data[self.data['station_id'] == station_id]
        row_loc = station_data.index.get_loc(start_idx)
        
        seq_len, pred_hor = self.config.seq_length, self.config.pred_horizon
        
        # Input and target sequences
        input_data = station_data.iloc[row_loc : row_loc + seq_len]
        target_data = station_data.iloc[row_loc + seq_len : row_loc + seq_len + pred_hor]
        
        # Extract features and target values
        feature_cols = ['total_energy', 'swell_energy', 'wind_energy', 'swell_fraction', 'wind_speed', 'wind_dir', 'wave_wind_alignment']
        features = input_data[feature_cols].values
        
        target_cols = ['significant_wave_height', 'peak_period', 'mean_wave_dir']
        target = target_data[target_cols].values
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'station_idx': torch.tensor(self.station_map[station_id], dtype=torch.long),
            'timestamps': input_data['timestamp'].values
        }