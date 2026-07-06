import gzip
import logging
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.wave_forecast.data.processor import VectorizedSpectrumProcessor

# Configure logging to see informative messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to run the historical data processing."""
    # Build a robust path to the config file
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    config_path = project_root / "config" / "production.yaml"

    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    processor = VectorizedSpectrumProcessor(config)
    all_processed_features = []
    
    archive_path = project_root / config["data_paths"]["historical_archive"]
    logging.info(f"Reading historical data from: {archive_path}")

    station_list = list(config["stations"]["coordinates"].keys())
    logging.info(f"Processing {len(station_list)} stations...")
    
    stations_processed = 0
    stations_with_data = 0
    stations_failed = 0
    
    for station_id in tqdm(station_list, desc="Processing Stations"):
        raw_data_content = {}
        file_types = ["wave", "spec", "raw_spectral", "alpha1", "alpha2", "r1", "r2"]
        
        # Load all data types for this station
        files_found = 0
        for data_type in file_types:
            file_path = archive_path / f"{station_id}_{data_type}.gz"
            if file_path.exists():
                try:
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        raw_data_content[data_type] = f.read()
                    files_found += 1
                except Exception as e:
                    logging.warning(f"Could not read {file_path}: {e}")
                    raw_data_content[data_type] = None
            else:
                raw_data_content[data_type] = None

        # Log data availability for troubleshooting
        if files_found == 0:
            logging.warning(f"No data files found for station {station_id}")
        
        stations_processed += 1
        
        try:
            features_df = processor.process(station_id, raw_data_content)
            if not features_df.empty:
                all_processed_features.append(features_df)
                stations_with_data += 1
                logging.debug(f"✅ {station_id}: {len(features_df)} features extracted")
            else:
                logging.debug(f"⚠️  {station_id}: No features extracted (likely missing raw_spectral data)")
        except Exception as e:
            stations_failed += 1
            logging.error(f"❌ Failed to process station {station_id}: {e}", exc_info=True)

    # Final results
    if not all_processed_features:
        logging.error("No features were processed for any station. Exiting.")
        logging.error("Check that your data files exist and are in the correct format.")
        return

    logging.info("Combining all processed features...")
    final_df = pd.concat(all_processed_features, ignore_index=True)
    
    output_path = project_root / config["data_paths"]["processed_features"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path)
    
    # Summary statistics
    logging.info(f"🌊 Historical processing complete!")
    logging.info(f"   - Stations processed: {stations_processed}")
    logging.info(f"   - Stations with valid features: {stations_with_data}")
    logging.info(f"   - Stations failed: {stations_failed}")
    logging.info(f"   - Total records processed: {len(final_df):,}")
    logging.info(f"   - Features saved to: {output_path}")
    
    # Data quality summary
    if len(final_df) > 0:
        stations_in_final = final_df['station_id'].nunique()
        avg_records_per_station = len(final_df) / stations_in_final
        logging.info(f"   - Average records per station: {avg_records_per_station:.0f}")
        
        # Check feature completeness
        feature_cols = [col for col in final_df.columns if col not in ['station_id', 'timestamp']]
        completeness = {}
        for col in feature_cols:
            if col in final_df.columns:
                non_null_pct = (1 - final_df[col].isnull().mean()) * 100
                completeness[col] = non_null_pct
        
        # Log the most important features
        key_features = ['significant_wave_height', 'total_energy', 'wind_speed', 'alpha1']
        for feature in key_features:
            if feature in completeness:
                logging.info(f"   - {feature}: {completeness[feature]:.1f}% complete")

if __name__ == "__main__":
    main()