import pandera as pa
from pandera import Column, Check
import numpy as np
import pandas as pd
from ..utils.config import load_config

# Load configuration to get the list of valid stations
config = load_config()
ACTIVE_STATIONS = list(config.stations.coordinates.keys())

class OceanDataSchema(pa.DataFrameModel):
    """Pandera schema for validating processed wave features."""
    
    # Metadata
    station_id: Column(str, checks=Check.isin(ACTIVE_STATIONS))
    timestamp: Column(pd.Timestamp)
        
    # Core wave parameters
    significant_wave_height: Column(float, checks=[Check.greater_than(0), Check.less_than(25)], nullable=True)
    peak_period: Column(float, checks=[Check.greater_than(1), Check.less_than(30)], nullable=True)
    mean_wave_dir: Column(float, checks=[Check.greater_than_or_equal_to(0), Check.less_than(360)], nullable=True)
        
    # Meteorological parameters
    wind_speed: Column(float, checks=[Check.greater_than_or_equal_to(0), Check.less_than(100)], nullable=True)
    wind_dir: Column(float, checks=[Check.greater_than_or_equal_to(0), Check.less_than(360)], nullable=True)
        
    # Spectral parameters
    total_energy: Column(float, checks=Check.greater_than_or_equal_to(0))
    swell_energy: Column(float, checks=Check.greater_than_or_equal_to(0))
    wind_energy: Column(float, checks=Check.greater_than_or_equal_to(0))
        
    # Physics-based cross-column checks
    @pa.check("significant_wave_height", "total_energy")
    def check_wave_height_energy_relation(cls, series: pd.Series, df: pd.DataFrame) -> bool:
        """Check Hs ≈ 4.004 * sqrt(total_energy) within tolerance."""
        valid_idx = ~df['total_energy'].isna() & ~df['significant_wave_height'].isna()
        if not valid_idx.any():
            return True
        
        hs_calculated = 4.004 * np.sqrt(df.loc[valid_idx, 'total_energy'])
        error = np.abs(df.loc[valid_idx, 'significant_wave_height'] - hs_calculated)
        tolerance = np.maximum(0.5, 0.1 * df.loc[valid_idx, 'significant_wave_height'])
        return (error <= tolerance).all()
        
    @pa.check("swell_energy", "wind_energy", "total_energy")
    def check_energy_components(cls, swell: pd.Series, wind: pd.Series, total: pd.Series) -> bool:
        """Check swell_energy + wind_energy ≈ total_energy with some tolerance."""
        valid_idx = ~swell.isna() & ~wind.isna() & ~total.isna()
        if not valid_idx.any():
            return True
            
        sum_components = swell.loc[valid_idx] + wind.loc[valid_idx]
        return (sum_components <= 1.05 * total.loc[valid_idx]).all()

    class Config:
        coerce = True  # Automatically coerce data types
        strict = False # Allow extra columns not defined in schema