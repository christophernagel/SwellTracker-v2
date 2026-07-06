import pandera as pa
from pandera.typing import Series
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# --- Build a robust path to the config file ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
config_path = project_root / "config" / "production.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

ACTIVE_STATIONS = list(config["stations"]["coordinates"].keys())


class OceanDataSchema(pa.DataFrameModel):
    """
    Pandera schema for validating processed wave features using modern,
    type-hinted syntax and robust, NaN-aware checks.
    """

    # --- Metadata Columns ---
    station_id: Series[str] = pa.Field(isin=ACTIVE_STATIONS, nullable=False)
    timestamp: Series[pd.Timestamp] = pa.Field(nullable=False)

    # --- Core Wave & Meteorological Parameters ---
    significant_wave_height: Series[float] = pa.Field(ge=0, le=25, nullable=True)
    peak_period: Series[float] = pa.Field(ge=1, le=30, nullable=True)
    mean_wave_dir: Series[float] = pa.Field(ge=0, le=360, nullable=True)
    wind_speed: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    wind_dir: Series[float] = pa.Field(ge=0, le=360, nullable=True)

    # --- Spectral & Directional Physics ---
    total_energy: Series[float] = pa.Field(ge=0, nullable=True)
    swell_energy: Series[float] = pa.Field(ge=0, nullable=True)
    wind_energy: Series[float] = pa.Field(ge=0, nullable=True)
    swell_fraction: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    alpha1: Series[float] = pa.Field(ge=0, le=360, nullable=True)
    alpha2: Series[float] = pa.Field(ge=0, le=360, nullable=True)
    r1: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    r2: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    bimodality: Series[float] = pa.Field(ge=-1, le=1, nullable=True)

    # --- Derived & Cross-Validation Features ---
    wave_wind_alignment: Series[float] = pa.Field(ge=0, le=180, nullable=True)
    energy_discrepancy: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        coerce = True
        strict = False 

    @pa.check("significant_wave_height", "total_energy", name="Hs vs. Energy")
    def check_wave_height_energy_relation(cls, df: pd.DataFrame) -> Series[bool]:
        """
        FIXED: Check Hs ≈ 4 * sqrt(total_energy) only where both values are present.
        If required columns are missing, the check passes to prevent KeyError.
        """
        # Check if required columns exist before accessing them
        if 'total_energy' not in df.columns or 'significant_wave_height' not in df.columns:
            return pd.Series(True, index=df.index)

        mask = (df['total_energy'].notna() & 
                df['significant_wave_height'].notna() & 
                (df['total_energy'] > 1e-6) & 
                (df['significant_wave_height'] > 1e-6))
        
        # If no rows have both values, the check passes by default
        if not mask.any():
            return pd.Series(True, index=df.index)

        hs_calculated = 4.004 * np.sqrt(df.loc[mask, 'total_energy'])
        hs_actual = df.loc[mask, 'significant_wave_height']
        
        relative_error = np.abs(hs_actual - hs_calculated) / hs_actual
        
        results = pd.Series(True, index=df.index)
        results.loc[mask] = relative_error <= 0.15  # 15% tolerance
        return results

    @pa.check("swell_energy", "wind_energy", "total_energy", name="Energy Components Sum")
    def check_energy_components(cls, df: pd.DataFrame) -> Series[bool]:
        """
        FIXED: Check if swell and wind energy sum to total energy within a 5% tolerance.
        If required columns are missing, the check passes to prevent KeyError.
        """
        # Check if required columns exist before accessing them
        required_cols = ['swell_energy', 'wind_energy', 'total_energy']
        if not all(col in df.columns for col in required_cols):
            return pd.Series(True, index=df.index)

        mask = (df['swell_energy'].notna() & 
                df['wind_energy'].notna() & 
                df['total_energy'].notna() & 
                (df['total_energy'] > 1e-6))
        
        if not mask.any():
            return pd.Series(True, index=df.index)
            
        sum_components = df.loc[mask, 'swell_energy'] + df.loc[mask, 'wind_energy']
        
        diff = np.abs(sum_components - df.loc[mask, 'total_energy'])
        tolerance = 0.05 * df.loc[mask, 'total_energy']
        
        results = pd.Series(True, index=df.index)
        results.loc[mask] = diff <= tolerance
        return results