#!/usr/bin/env python3
import argparse
import gzip
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.wave_forecast.data.processor import VectorizedSpectrumProcessor
from src.wave_forecast.utils.config import load_config
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def group_historical_files(input_dir: Path) -> dict:
    """Groups historical gzipped files by station ID."""
    station_files = defaultdict(dict)
    logging.info(f"Scanning for historical files in {input_dir}...")
    
    for gz_file in input_dir.glob('*.gz'):
        parts = gz_file.name.split('_')
        if len(parts) < 2:
            continue
        
        station_id = parts[0]
        
        # Revised mapping to match the new processor's expected keys
        if 'raw_spectral' in gz_file.name:
            data_type = 'raw_spectral'
        elif 'directional_alpha1' in gz_file.name:
            data_type = 'alpha1'
        elif 'directional_alpha2' in gz_file.name:
            data_type = 'alpha2'
        elif 'directional_r1' in gz_file.name:
            data_type = 'r1'
        elif 'directional_r2' in gz_file.name:
            data_type = 'r2'
        elif 'spectral' in gz_file.name: # This is the summary .spec file
            data_type = 'spec'
        elif 'wave' in gz_file.name: # This is the met/wave summary file
            data_type = 'wave'
        else:
            continue
            
        station_files[station_id][data_type] = gz_file
        
    logging.info(f"Found data for {len(station_files)} stations.")
    return station_files

def main(args):
    """Main function to run the historical data backfill process."""
    config = load_config(args.config)
    station_file_groups = group_historical_files(Path(args.input_dir))
    
    # Initialize the processor with the centralized configuration
    processor = VectorizedSpectrumProcessor(config)
    
    all_processed_features = []
    
    # Use tqdm for a progress bar
    for station_id, data_type_map in tqdm(station_file_groups.items(), desc="Processing Stations"):
        logging.info(f"Processing station: {station_id}")
        
        raw_data_content = {}
        for data_type, file_path in data_type_map.items():
            try:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    raw_data_content[data_type] = f.read()
            except Exception as e:
                logging.warning(f"Could not read {file_path}: {e}")
                raw_data_content[data_type] = None

        try:
            # Process all data for the station
            features_df = processor.process(station_id, raw_data_content)
            if not features_df.empty:
                all_processed_features.append(features_df)
        except Exception as e:
            logging.error(f"Failed to process station {station_id}: {e}", exc_info=True)

    if not all_processed_features:
        logging.error("No features were processed. Exiting.")
        return

    # Combine all dataframes into one and save
    logging.info("Combining all processed features...")
    final_df = pd.concat(all_processed_features, ignore_index=True)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path)
    
    logging.info(f"✅ Historical backfill complete! Features saved to: {output_path}")
    logging.info(f"Total records processed: {len(final_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process historical NDBC data archives.")
    parser.add_argument("--input-dir", default="data/historical_archive", help="Directory containing the historical .gz files.")
    parser.add_argument("--output-file", default="data/processed/historical_features.parquet", help="Path to save the final Parquet feature file.")
    parser.add_argument("--config", default="config/production.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    main(args)