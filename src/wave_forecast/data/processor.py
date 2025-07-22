import numpy as np
import pandas as pd
import io
from ..utils.physics import calculate_spectral_moments, separate_swell_wind
from ..utils.numerics import safe_divide
from .validation import OceanDataSchema

class VectorizedSpectrumProcessor:
    """Processes raw NDBC data files into a validated, feature-rich DataFrame."""
    def __init__(self, config):
        self.config = config.processor
        self.freq_bins = np.array(self.config.freq_bins)
        self.freq_centers = (self.freq_bins[:-1] + self.freq_bins[1:]) / 2
        self.max_gap = pd.Timedelta(self.config.max_time_gap)

    def process(self, station_id, raw_data):
        # Step 1: Parse all raw data types into DataFrames
        data_spec_df = self._parse_spectral(raw_data.get('data_spec'))
        txt_df = self._parse_txt(raw_data.get('txt'))
        spec_df = self._parse_spec(raw_data.get('spec'))
        
        directional_dfs = {
            dtype: self._parse_directional(raw_data.get(dtype), dtype)
            for dtype in ['alpha1', 'alpha2', 'r1', 'r2']
        }

        # Step 2: Merge all data sources
        all_dfs = {'data_spec': data_spec_df, 'txt': txt_df, 'spec': spec_df, **directional_dfs}
        merged_df = self._merge_data(all_dfs)

        if merged_df.empty:
            return pd.DataFrame()

        # Step 3: Calculate physics-informed features
        features_df = self._calculate_features_vectorized(merged_df)

        # Step 4: Add metadata and validate
        features_df['station_id'] = station_id
        features_df.reset_index(inplace=True)
        return OceanDataSchema.validate(features_df)
    
    def _parse_txt(self, content):
        if not content: return pd.DataFrame()
        col_names = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']
        try:
            df = pd.read_csv(io.StringIO(content), delim_whitespace=True, skiprows=2, names=col_names, na_values=['MM', '99.0', '999.0', '9999.0'])
            df['timestamp'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))
            return df.set_index('timestamp')[['WDIR', 'WSPD']].rename(columns={'WDIR': 'wind_dir', 'WSPD': 'wind_speed'})
        except (ValueError, IndexError): return pd.DataFrame()

    def _parse_spec(self, content):
        if not content: return pd.DataFrame()
        col_names = ['YY', 'MM', 'DD', 'hh', 'mm', 'WVHT', 'SwH', 'SwP', 'WWH', 'WWP', 'SwD', 'WWD', 'STEEPNESS', 'APD', 'MWD']
        try:
            df = pd.read_csv(io.StringIO(content), delim_whitespace=True, skiprows=2, names=col_names, na_values=['MM', '99.0', '999.0', '9999.0'])
            df['timestamp'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))
            return df.set_index('timestamp')[['WVHT', 'DPD', 'MWD', 'APD']].rename(columns={'WVHT': 'significant_wave_height', 'DPD': 'peak_period', 'MWD': 'mean_wave_dir', 'APD': 'avg_wave_period'})
        except (ValueError, IndexError): return pd.DataFrame()
    
    def _parse_spectral(self, content):
        if not content: return pd.DataFrame()
        lines = content.strip().split('\n')
        if len(lines) < 2: return pd.DataFrame()
        try:
            df = pd.read_csv(io.StringIO('\n'.join(lines[1:])), delim_whitespace=True, header=None)
            df['timestamp'] = pd.to_datetime(df[[0, 1, 2, 3, 4]].rename(columns={0:'year', 1:'month', 2:'day', 3:'hour', 4:'minute'}))
            spectral_data = df.iloc[:, 5:5+len(self.freq_bins)-1]
            spectral_data.columns = [f"freq_{f:.5f}" for f in self.freq_centers]
            spectral_data.index = df['timestamp']
            return spectral_data
        except (ValueError, IndexError): return pd.DataFrame()

    def _parse_directional(self, content, dtype):
        if not content: return pd.DataFrame()
        col_names = ['YY', 'MM', 'DD', 'hh', 'mm', dtype]
        try:
            df = pd.read_csv(io.StringIO(content), delim_whitespace=True, skiprows=2, names=col_names, usecols=[0, 1, 2, 3, 4, 6], na_values=['MM', '999.0'])
            df['timestamp'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))
            return df.set_index('timestamp')[[dtype]]
        except (ValueError, IndexError): return pd.DataFrame()

    def _merge_data(self, data_dfs):
        base_df = data_dfs.get('data_spec')
        if base_df is None or base_df.empty: return pd.DataFrame()
        merged = base_df.copy()
        for name, df in data_dfs.items():
            if name == 'data_spec' or df.empty: continue
            df_renamed = df.rename(columns={col: f"{name}_{col}" for col in df.columns})
            merged = pd.merge_asof(merged.sort_index(), df_renamed.sort_index(), left_index=True, right_index=True, direction='nearest', tolerance=self.max_gap)
        return merged

    def _calculate_features_vectorized(self, df):
        spec_cols = [col for col in df.columns if col.startswith('freq_')]
        spectra = df[spec_cols].values
        
        # --- Spectral Physics ---
        df['total_energy'] = calculate_spectral_moments(spectra, self.freq_centers, moment=0)
        wind_speeds = df['txt_wind_speed'].fillna(0).values
        swell_energy, wind_energy = separate_swell_wind(spectra, self.freq_centers, wind_speeds)
        df['swell_energy'] = swell_energy
        df['wind_energy'] = wind_energy
        df['swell_fraction'] = safe_divide(df['swell_energy'], df['total_energy'])

        # --- Directional Physics ---
        if 'spec_r1' in df and 'spec_r2' in df:
            df['bimodality'] = df['spec_r1'] + df['spec_r2'] - 1.0

        # --- Meteorological Physics ---
        if 'txt_wind_dir' in df and 'spec_mean_wave_dir' in df:
            diff = np.abs(df['spec_mean_wave_dir'] - df['txt_wind_dir'])
            df['wave_wind_alignment'] = np.minimum(diff, 360 - diff)

        # --- Cross-Validation Features ---
        if 'spec_significant_wave_height' in df:
            expected_energy = (df['spec_significant_wave_height'] ** 2) / 16.0
            df['energy_discrepancy'] = np.abs(df['total_energy'] - expected_energy)

        # Final feature selection
        final_feature_columns = ['spec_significant_wave_height', 'spec_peak_period', 'spec_mean_wave_dir', 'total_energy', 'swell_energy', 'wind_energy', 'swell_fraction', 'txt_wind_speed', 'txt_wind_dir', 'wave_wind_alignment', 'energy_discrepancy']
        
        for col in final_feature_columns:
            if col not in df: df[col] = np.nan
        
        return df.dropna(subset=['total_energy'])