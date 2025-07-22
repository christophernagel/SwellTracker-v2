#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.wave_forecast.utils.config import load_config
from src.wave_forecast.data.sequencer import WaveSequenceDataset
from src.wave_forecast.model.architecture import SpatiotemporalTransformer
from src.wave_forecast.model.loss import PhysicsInformedLoss
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class WaveForecastLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for the wave forecasting model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SpatiotemporalTransformer(config)
        self.loss_fn = PhysicsInformedLoss()
        self.save_hyperparameters(config)

    def forward(self, x, station_idx):
        return self.model(x, station_idx)

    def training_step(self, batch, batch_idx):
        features, target, station_idx, _ = batch.values()
        prediction = self(features, station_idx)
        loss = self.loss_fn(prediction, target, features)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, target, station_idx, _ = batch.values()
        prediction = self(features, station_idx)
        loss = self.loss_fn(prediction, target, features)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def main(args):
    # 1. Load Config and Data
    config = load_config(args.config)
    try:
        features_df = pd.read_parquet(f"{config.data_paths.processed}/features_latest.parquet")
    except FileNotFoundError:
        print(f"Error: Processed features not found. Run the collection pipeline first.")
        return

    # 2. Create Datasets with time-based splits (Corrected)
    print("Creating time-based training and validation splits...")
    train_dataset = WaveSequenceDataset(features_df, config, split='train')
    val_dataset = WaveSequenceDataset(features_df, config, split='val')
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    # 3. Initialize Model and Trainer
    model = WaveForecastLightning(config)

    # (Optional) Initialize W&B Logger
    # wandb_logger = WandbLogger(project="swelltracker-v2", config=OmegaConf.to_container(config, resolve=True))
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.data_paths.models,
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=False,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=50
    )

    # 4. Run Training
    print("Starting model training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SwellTracker wave forecasting model.")
    parser.add_argument("--config", default="config/production.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    main(args)