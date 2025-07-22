#!/usr/bin/env python3
import argparse
from src.wave_forecast.mlops.orchestration import data_pipeline_flow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SwellTracker data collection and processing pipeline.")
    parser.add_argument("--config", default="config/production.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    
    data_pipeline_flow(config_path=args.config)