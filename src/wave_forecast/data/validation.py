import pandera as pa
from pandera import Column, Check
import numpy as np
import pandas as pd
from ..utils.config import load_config

# Load configuration to get the list of valid stations
config = load_config()
ACTIVE_STATIONS = list(config.stations.coordinates.keys())


class OceanDataSchema(pa.DataFrameModel):
    """Pandera schema for validating processed wave features using correct Field syntax."""

    # --- Metadata Columns ---
    station_id: pa.typing.Series[str] = pa.Field(isin=ACTIVE_STATIONS)
    timestamp: pa.typing.Series[pd.Timestamp] = pa.Field()

    # --- Core Wave Parameters ---
    significant_wave_height: pa.typing.Series[float] = pa.Field(gt=0, lt=25, nullable=True)
    peak_period: pa.typing.Series[float] = pa.Field(gt=1, lt=30, nullable=True)
    mean_wave_dir: pa.typing.Series[float] = pa.Field(ge=0, lt=360, nullable=True)

    # --- Meteorological Parameters ---
    wind_speed: pa.typing.Series[float] = pa.Field(ge=0, lt=100, nullable=True)
    wind_dir: pa.typing.Series[float] = pa.Field(ge=0, lt=360, nullable=True)

    # --- Spectral Physics Parameters ---
    total_energy: pa.typing.Series[float] = pa.Field(ge=0, nullable=True)
    swell_energy: pa.typing.Series[float] = pa.Field(ge=0, nullable=True)
    wind_energy: pa.typing.Series[float] = pa.Field(ge=0, nullable=True)
    swell_fraction: pa.typing.Series[float] = pa.Field(ge=0, le=1, nullable=True)

    # --- Directional Physics Parameters ---
    alpha1: pa.typing.Series[float] = pa.Field(ge=0, le=360, nullable=True)
    alpha2: pa.typing.Series[float] = pa.Field(ge=0, le=360, nullable=True)
    r1: pa.typing.Series[float] = pa.Field(ge=0, le=1, nullable=True)
    r2: pa.typing.Series[float] = pa.Field(ge=0, le=1, nullable=True)
    bimodality: pa.typing.Series[float] = pa.Field(nullable=True)

    # --- Cross-Validation & Derived Features ---
    wave_wind_alignment: pa.typing.Series[float] = pa.Field(ge=0, le=180, nullable=True)
    energy_discrepancy: pa.typing.Series[float] = pa.Field(ge=0, nullable=True)


    class Config:
        """Pandera configuration."""
        coerce = True
        strict = False

    # --- Physics-Based Cross-Column Checks ---
    @pa.check("significant_wave_height", "total_energy", name="Hs vs. Energy")
    def check_wave_height_energy_relation(cls, df: pd.DataFrame) -> bool:
        """Check Hs ≈ 4.004 * sqrt(total_energy) within tolerance."""
        valid_idx = df['total_energy'].notna() & df['significant_wave_height'].notna()
        if not valid_idx.any():
            return True

        hs_calculated = 4.004 * np.sqrt(df.loc[valid_idx, 'total_energy'])
        error = np.abs(df.loc[valid_idx, 'significant_wave_height'] - hs_calculated)
        tolerance = np.maximum(0.5, 0.1 * df.loc[valid_idx, 'significant_wave_height'])
        return (error <= tolerance).all()

    @pa.check("swell_energy", "wind_energy", "total_energy", name="Energy Components Sum")
    def check_energy_components(cls, df: pd.DataFrame) -> bool:
        """Check swell_energy + wind_energy ≈ total_energy with some tolerance."""
        valid_idx = df['swell_energy'].notna() & df['wind_energy'].notna() & df['total_energy'].notna()
        if not valid_idx.any():
            return True
            
        sum_components = df.loc[valid_idx, 'swell_energy'] + df.loc[valid_idx, 'wind_energy']
        return (sum_components <= 1.05 * df.loc[valid_idx, 'total_energy']).all()