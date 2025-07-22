from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
from ..data.collector import RobustBuoyCollector # Assumes this exists
from ..data.processor import VectorizedSpectrumProcessor
from ..utils.config import load_config
import logging

logger = logging.getLogger(__name__)

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1), retries=3)
def collect_station_data(station_id, data_type, config):
    collector = RobustBuoyCollector(config) # Assumes collector takes config
    return collector.collect(station_id, data_type)

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=7))
def process_station_data(station_id, raw_data, config):
    processor = VectorizedSpectrumProcessor(config)
    return processor.process(station_id, raw_data)

@flow(name="SwellTracker-Data-Pipeline")
def data_pipeline_flow(config_path="config/production.yaml"):
    config = load_config(config_path)
    stations = config.stations.active
    
    all_features = []
    for station_id in stations:
        try:
            raw_data = {
                dtype: collect_station_data(station_id, dtype, config)
                for dtype in ['spec', 'txt', 'data_spec', 'alpha1', 'alpha2', 'r1', 'r2']
            }
            features = process_station_data(station_id, raw_data, config)
            all_features.append(features)
        except Exception as e:
            logger.error(f"Failed processing station {station_id}: {e}")
            
    # Combine and save to feature store
    final_df = pd.concat(all_features, ignore_index=True)
    feature_store_path = f"{config.data_paths.processed}/features_latest.parquet"
    final_df.to_parquet(feature_store_path)
    logger.info(f"Data pipeline complete. Features saved to {feature_store_path}")
    
    return feature_store_path