import numpy as np
import pandas as pd
import io
import logging
from ..utils.physics import calculate_spectral_moments, separate_swell_wind
from ..utils.numerics import safe_divide
from .validation import OceanDataSchema

class VectorizedSpectrumProcessor:
    """
    Processes raw NDBC data files into a validated, feature-rich DataFrame
    using fully vectorized operations and robust parsers tailored to the exact file formats.
    """
    def __init__(self, config: dict):
        self.processor_config = config["processor"]
        self.target_freqs = np.array(self.processor_config["freq_bins"])
        self.target_freq_centers = (self.target_freqs[:-1] + self.target_freqs[1:]) / 2
        self.max_gap = pd.Timedelta(self.processor_config["max_time_gap"])

    def process(self, station_id: str, raw_data: dict) -> pd.DataFrame:
        """Main processing function with robust error handling."""
        # Step 1: Parse all raw data types
        spec_df = self._parse_spec(raw_data.get('spec'))
        wave_df = self._parse_wave(raw_data.get('wave'))
        raw_spectral_df = self._parse_raw_spectral(raw_data.get('raw_spectral'))
        
        directional_dfs = {
            dtype: self._parse_directional(raw_data.get(dtype), dtype)
            for dtype in ['alpha1', 'alpha2', 'r1', 'r2']
        }

        # Step 1.5: Interpolate spectra to standard grid
        if not raw_spectral_df.empty:
            raw_spectral_df = self._interpolate_spectra(raw_spectral_df)

        # Step 2: Merge all data sources
        all_dfs = {
            'spec': spec_df, 'wave': wave_df, 'raw_spectral': raw_spectral_df, **directional_dfs
        }
        merged_df = self._merge_data(all_dfs)

        if merged_df.empty:
            logging.warning(f"No data could be merged for station {station_id}. Skipping.")
            return pd.DataFrame()

        # Step 3: Calculate physics-informed features
        features_df = self._calculate_features_vectorized(merged_df)

        if features_df.empty:
            logging.warning(f"No valid features could be calculated for station {station_id}. Skipping.")
            return pd.DataFrame()

        # Step 4: Add metadata and validate
        features_df['station_id'] = station_id
        features_df.reset_index(inplace=True)
        
        try:
            return OceanDataSchema.validate(features_df)
        except Exception as e:
            # Provide more detailed error info for debugging validation
            logging.error(f"Final validation failed for station {station_id}: {e}", exc_info=True)
            return pd.DataFrame()

    def _parse_wave(self, content: str) -> pd.DataFrame:
        """Parses wave files containing DPD and meteorological data."""
        if not content: 
            return pd.DataFrame()
        
        lines = [line for line in content.strip().split('\n') if not line.startswith('#')]
        if not lines:
            return pd.DataFrame()
            
        col_names = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'PTDY', 'TIDE']
        try:
            df = pd.read_csv(io.StringIO('\n'.join(lines)), sep='\s+', names=col_names, na_values=['MM', '99.0', '999.0', '9999.0'])
            df['timestamp'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))
            df = df.set_index('timestamp')
            
            # Select and rename only available columns to avoid KeyErrors
            rename_map = {'WDIR': 'wind_dir', 'WSPD': 'wind_speed', 'DPD': 'peak_period'}
            final_cols = {k: v for k, v in rename_map.items() if k in df.columns}
            if not final_cols: 
                return pd.DataFrame()
            
            return df[list(final_cols.keys())].rename(columns=final_cols)
        except Exception as e:
            logging.warning(f"Could not parse wave file: {e}")
            return pd.DataFrame()

    def _parse_spec(self, content: str) -> pd.DataFrame:
        """Parses spectral summary files."""
        if not content: 
            return pd.DataFrame()

        lines = [line for line in content.strip().split('\n') if not line.startswith('#')]
        if not lines:
            return pd.DataFrame()

        col_names = ['YY', 'MM', 'DD', 'hh', 'mm', 'WVHT', 'SwH', 'SwP', 'WWH', 'WWP', 'SwD', 'WWD', 'STEEPNESS', 'APD', 'MWD']
        try:
            df = pd.read_csv(io.StringIO('\n'.join(lines)), sep='\s+', names=col_names, na_values=['MM', '99.0', '999.0', '9999.0'])
            df['timestamp'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))
            df = df.set_index('timestamp')

            rename_map = {'WVHT': 'significant_wave_height', 'APD': 'avg_wave_period', 'MWD': 'mean_wave_dir'}
            final_cols = {k: v for k, v in rename_map.items() if k in df.columns}
            if not final_cols: 
                return pd.DataFrame()

            return df[list(final_cols.keys())].rename(columns=final_cols)
        except Exception as e:
            logging.warning(f"Could not parse spec file: {e}")
            return pd.DataFrame()

    def _parse_raw_spectral(self, content: str) -> pd.DataFrame:
        """Parser for raw spectral energy density files."""
        if not content: 
            return pd.DataFrame()
        
        lines = content.strip().split('\n')
        if len(lines) < 1: 
            return pd.DataFrame()
        
        data, original_freqs = [], None
        for line in lines:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 8: continue
            
            try:
                year, month, day, hour, minute = map(int, parts[0:5])
                if year < 100: year += 2000
                dt = pd.to_datetime(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}")

                spectral_vals, frequencies = [], []
                i = 6
                while i < len(parts) - 1:
                    spec_str, freq_str = parts[i], parts[i+1]
                    if '(' in freq_str and ')' in freq_str:
                        freq = float(freq_str.strip('()'))
                        if spec_str not in ['MM', '999.00', '99.00']:
                            spectral_vals.append(float(spec_str))
                            frequencies.append(freq)
                    i += 2
                
                if original_freqs is None and frequencies:
                    original_freqs = frequencies
                if len(spectral_vals) == len(original_freqs or []):
                    data.append([dt] + spectral_vals)
            except Exception:
                continue
        
        if not data or original_freqs is None: 
            return pd.DataFrame()
        
        columns = ['timestamp'] + [f"raw_spec_{i}" for i in range(len(original_freqs))]
        df = pd.DataFrame(data, columns=columns).set_index('timestamp')
        df.attrs['original_frequencies'] = np.array(original_freqs)
        return df

    def _parse_directional(self, content: str, dtype: str) -> pd.DataFrame:
        """Parser for directional wave data files (alpha1, alpha2, r1, r2)."""
        if not content: 
            return pd.DataFrame()
        
        lines = content.strip().split('\n')
        if len(lines) < 1: 
            return pd.DataFrame()
        
        data = []
        for line in lines:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 7: continue
            
            try:
                year, month, day, hour, minute = map(int, parts[0:5])
                if year < 100: year += 2000
                dt = pd.to_datetime(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}")

                dir_vals = []
                i = 5
                while i < len(parts) - 1:
                    val_str, freq_str = parts[i], parts[i+1]
                    if '(' in freq_str and ')' in freq_str and val_str not in ['MM', '999.0', '999.00']:
                        dir_vals.append(float(val_str))
                    i += 2
                
                if dir_vals:
                    data.append([dt, np.mean(dir_vals)])
            except Exception:
                continue
        
        if not data: 
            return pd.DataFrame()
        return pd.DataFrame(data, columns=['timestamp', dtype]).set_index('timestamp')

    def _interpolate_spectra(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolates raw spectra to the standard target frequency grid."""
        original_freqs = df.attrs.get('original_frequencies')
        if original_freqs is None: 
            return pd.DataFrame()
            
        interpolated_spectra = np.apply_along_axis(
            lambda spectrum: np.interp(self.target_freq_centers, original_freqs, spectrum, left=0, right=0),
            axis=1, arr=df.values
        )
        
        new_columns = [f"freq_{f:.5f}" for f in self.target_freq_centers]
        return pd.DataFrame(interpolated_spectra, index=df.index, columns=new_columns)

    def _merge_data(self, data_dfs: dict) -> pd.DataFrame:
        """Merges multiple dataframes, correctly handling column name prefixes."""
        # Use raw_spectral as the primary base for merging
        base_df = data_dfs.get('raw_spectral')
        base_name = 'raw_spectral'
        
        # Fallback to 'spec' if 'raw_spectral' is unavailable
        if base_df is None or base_df.empty:
            base_df = data_dfs.get('spec')
            base_name = 'spec'
            if base_df is None or base_df.empty:
                return pd.DataFrame()

        merged = base_df.copy()
        for name, df in data_dfs.items():
            if name == base_name or df.empty:
                continue

            # Only prefix non-directional data sources
            if name in ['alpha1', 'alpha2', 'r1', 'r2']:
                df_to_merge = df
            else:
                df_to_merge = df.rename(columns={col: f"{name}_{col}" for col in df.columns})

            merged = pd.merge_asof(
                merged.sort_index(), df_to_merge.sort_index(),
                left_index=True, right_index=True,
                direction='nearest', tolerance=self.max_gap
            )
        return merged

    def _calculate_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED: Vectorized feature calculation that guarantees all schema columns
        exist before validation, preventing KeyErrors.
        """
        # --- 1. Initialize a DataFrame with all expected columns to guarantee structure ---
        schema_columns = [
            col for col in OceanDataSchema.to_schema().columns.keys() 
            if col not in ['station_id', 'timestamp']
        ]
        final_df = pd.DataFrame(index=df.index, columns=schema_columns)
        final_df = final_df.astype(float)  # Ensure all columns are float type for calculations

        # --- 2. Fill columns directly from the merged DataFrame ---
        final_df['significant_wave_height'] = df.get('significant_wave_height')
        # If the primary source is 'spec', it won't have the 'spec_' prefix
        if 'spec_significant_wave_height' in df.columns:
            final_df['significant_wave_height'].fillna(df['spec_significant_wave_height'], inplace=True)
            
        final_df['peak_period'] = df.get('wave_peak_period')
        final_df['mean_wave_dir'] = df.get('mean_wave_dir')
        if 'spec_mean_wave_dir' in df.columns:
            final_df['mean_wave_dir'].fillna(df['spec_mean_wave_dir'], inplace=True)
            
        final_df['wind_speed'] = df.get('wave_wind_speed')
        final_df['wind_dir'] = df.get('wave_wind_dir')
        final_df['alpha1'] = df.get('alpha1')
        final_df['alpha2'] = df.get('alpha2')
        final_df['r1'] = df.get('r1')
        final_df['r2'] = df.get('r2')
        
        # --- 3. Conditionally calculate physics-based features IF spectral data exists ---
        spec_cols = [col for col in df.columns if col.startswith('freq_')]
        if spec_cols:
            spectra = df[spec_cols].values
            final_df['total_energy'] = calculate_spectral_moments(spectra, self.target_freq_centers, 0)
            
            wind_speeds = final_df['wind_speed'].fillna(0).values
            swell_energy, wind_energy = separate_swell_wind(spectra, self.target_freq_centers, wind_speeds)
            
            final_df['swell_energy'] = swell_energy
            final_df['wind_energy'] = wind_energy
            final_df['swell_fraction'] = safe_divide(swell_energy, final_df['total_energy'])

        # --- 4. Calculate derived features from the columns that now exist in final_df ---
        if 'r1' in final_df.columns and 'r2' in final_df.columns:
            final_df['bimodality'] = final_df['r1'] + final_df['r2'] - 1.0

        if 'wind_dir' in final_df.columns and 'mean_wave_dir' in final_df.columns:
            diff = np.abs(final_df['mean_wave_dir'] - final_df['wind_dir'])
            final_df['wave_wind_alignment'] = np.minimum(diff, 360 - diff)

        # This check is now safe because 'total_energy' column is guaranteed to exist
        if final_df['significant_wave_height'].notna().any() and final_df['total_energy'].notna().any():
            mask = final_df['significant_wave_height'].notna() & final_df['total_energy'].notna()
            expected_energy = (final_df.loc[mask, 'significant_wave_height'] ** 2) / 16.0
            final_df.loc[mask, 'energy_discrepancy'] = np.abs(final_df.loc[mask, 'total_energy'] - expected_energy)
        
        # --- 5. Drop rows where essential data (total_energy) could not be calculated ---
        return final_df.dropna(subset=['total_energy'], how='all')