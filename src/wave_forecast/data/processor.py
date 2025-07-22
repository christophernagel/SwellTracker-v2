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
    def __init__(self, config):
        self.config = config.processor
        self.target_freqs = np.array(self.config.freq_bins)
        self.target_freq_centers = (self.target_freqs[:-1] + self.target_freqs[1:]) / 2
        self.max_gap = pd.Timedelta(self.config.max_time_gap)

    def process(self, station_id: str, raw_data: dict) -> pd.DataFrame:
        """Main processing function"""
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
            return pd.DataFrame()

        # Step 3: Calculate physics-informed features
        features_df = self._calculate_features_vectorized(merged_df)

        # Step 4: Add metadata and validate
        features_df['station_id'] = station_id
        features_df.reset_index(inplace=True)
        return OceanDataSchema.validate(features_df)

    def _parse_wave(self, content: str) -> pd.DataFrame:
        """Parses wave files containing DPD and meteorological data."""
        if not content: 
            return pd.DataFrame()
        
        col_names = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'PTDY', 'TIDE']
        try:
            df = pd.read_csv(io.StringIO(content), sep='\s+', skiprows=2, names=col_names, na_values=['MM', '99.0', '999.0', '9999.0'])
            df['timestamp'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))
            return df.set_index('timestamp')[['WDIR', 'WSPD', 'DPD']].rename(columns={'WDIR': 'wind_dir', 'WSPD': 'wind_speed', 'DPD': 'peak_period'})
        except (ValueError, IndexError): 
            return pd.DataFrame()

    def _parse_spec(self, content: str) -> pd.DataFrame:
        """Parses spectral summary files."""
        if not content: 
            return pd.DataFrame()
        
        col_names = ['YY', 'MM', 'DD', 'hh', 'mm', 'WVHT', 'SwH', 'SwP', 'WWH', 'WWP', 'SwD', 'WWD', 'STEEPNESS', 'APD', 'MWD']
        try:
            df = pd.read_csv(io.StringIO(content), sep='\s+', skiprows=2, names=col_names, na_values=['MM', '99.0', '999.0', '9999.0'])
            df['timestamp'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))
            column_map = {'WVHT': 'significant_wave_height', 'APD': 'avg_wave_period', 'MWD': 'mean_wave_dir'}
            available_cols = [col for col in column_map.keys() if col in df.columns]
            if not available_cols: 
                return pd.DataFrame()
            return df.set_index('timestamp')[available_cols].rename(columns=column_map)
        except (ValueError, IndexError): 
            return pd.DataFrame()

    def _parse_raw_spectral(self, content: str) -> pd.DataFrame:
        """FIXED: Parser matching your original data format"""
        if not content: 
            return pd.DataFrame()
        
        lines = content.strip().split('\n')
        if len(lines) < 1: 
            return pd.DataFrame()
        
        data = []
        original_freqs = None
        
        for line in lines:
            if line.startswith('#'): 
                continue
            
            parts = line.split()
            if len(parts) < 8:  # 5 time + 1 sep_freq + at least 2 spectral values
                continue
                
            try:
                # Parse timestamp (matching your original format)
                year = int(parts[0])
                if year < 100:
                    year += 2000
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                
                # Validate timestamp
                if not (1900 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
                    continue
                    
                dt = pd.to_datetime(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}")
                
                # Skip sep_freq (parts[5]) and parse spectral values starting at parts[6]
                spectral_vals = []
                frequencies = []
                
                i = 6
                while i < len(parts) - 1:
                    try:
                        spec_str = parts[i]
                        freq_str = parts[i + 1]
                        
                        # Handle the (frequency) format: "0.000 (0.033)"
                        if '(' in freq_str and ')' in freq_str:
                            freq = float(freq_str.strip('()'))
                            
                            # Check for missing data indicators
                            if spec_str not in ['MM', '999.00', '-999.00', '99.00']:
                                spec_val = float(spec_str)
                                
                                # Basic validation
                                if 0.020 <= freq <= 0.500 and 0 <= spec_val <= 100:
                                    spectral_vals.append(spec_val)
                                    frequencies.append(freq)
                        
                        i += 2  # Move to next value/frequency pair
                    except (ValueError, IndexError):
                        i += 1
                        continue
                
                # Set frequencies from first valid line
                if original_freqs is None and frequencies:
                    original_freqs = frequencies
                    logging.info(f"Set original frequencies: {len(frequencies)} bins from {min(frequencies):.3f} to {max(frequencies):.3f} Hz")
                
                # Only keep lines with consistent frequency count
                if len(spectral_vals) == len(original_freqs):
                    data.append([dt] + spectral_vals)
                    
            except Exception as e:
                logging.debug(f"Skipping raw spectral line: {str(e)}")
                continue
        
        if not data or original_freqs is None:
            logging.warning("No valid spectral data found")
            return pd.DataFrame()
        
        # Create DataFrame
        columns = ['timestamp'] + [f"raw_spec_{i}" for i in range(len(original_freqs))]
        df = pd.DataFrame(data, columns=columns).set_index('timestamp')
        df.attrs['original_frequencies'] = np.array(original_freqs)
        
        logging.info(f"Parsed {len(data)} spectral records with {len(original_freqs)} frequency bins")
        return df

    def _parse_directional(self, content: str, dtype: str) -> pd.DataFrame:
        """FIXED: Directional parser matching your original format"""
        if not content: 
            return pd.DataFrame()
        
        lines = content.strip().split('\n')
        if len(lines) < 1: 
            return pd.DataFrame()
        
        data = []
        
        for line in lines:
            if line.startswith('#'): 
                continue
                
            parts = line.split()
            if len(parts) < 7:  # 5 time + at least 1 value/freq pair
                continue
                
            try:
                # Parse timestamp (same format as spectral)
                year = int(parts[0])
                if year < 100:
                    year += 2000
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                
                if not (1900 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
                    continue
                    
                dt = pd.to_datetime(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}")
                
                # Parse directional values starting at parts[5] (no sep_freq column)
                dir_vals = []
                i = 5
                while i < len(parts) - 1:
                    try:
                        val_str = parts[i]
                        freq_str = parts[i + 1]
                        
                        if '(' in freq_str and ')' in freq_str:
                            freq = float(freq_str.strip('()'))
                            
                            if val_str not in ['999.0', '999.00', 'MM']:
                                val = float(val_str)
                                
                                # Validate based on data type
                                if dtype in ['alpha1', 'alpha2'] and 0 <= val <= 360:
                                    dir_vals.append(val)
                                elif dtype in ['r1', 'r2'] and 0 <= val <= 1:
                                    dir_vals.append(val)
                        
                        i += 2
                    except (ValueError, IndexError):
                        i += 1
                        continue
                
                if dir_vals:  # Take mean of all frequency-dependent values
                    mean_val = np.mean(dir_vals)
                    data.append([dt, mean_val])
                    
            except Exception as e:
                logging.debug(f"Skipping directional line: {str(e)}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=['timestamp', dtype])
        return df.set_index('timestamp')

    def _interpolate_spectra(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolates spectra to target frequency grid"""
        original_freqs = df.attrs.get('original_frequencies')
        if original_freqs is None: 
            return pd.DataFrame()

        raw_spectra_values = df.values
        
        # Interpolate each spectrum to target grid
        interpolated_spectra = np.apply_along_axis(
            lambda spectrum: np.interp(self.target_freq_centers, original_freqs, spectrum),
            axis=1, arr=raw_spectra_values
        )
        
        new_columns = [f"freq_{f:.5f}" for f in self.target_freq_centers]
        return pd.DataFrame(interpolated_spectra, index=df.index, columns=new_columns)

    def _merge_data(self, data_dfs: dict) -> pd.DataFrame:
        """Merges multiple dataframes"""
        base_df = data_dfs.get('raw_spectral')
        if base_df is None or base_df.empty: 
            return pd.DataFrame()
            
        merged = base_df.copy()
        for name, df in data_dfs.items():
            if name == 'raw_spectral' or df.empty: 
                continue
            df_renamed = df.rename(columns={col: f"{name}_{col}" for col in df.columns})
            merged = pd.merge_asof(
                merged.sort_index(), df_renamed.sort_index(), 
                left_index=True, right_index=True, 
                direction='nearest', tolerance=self.max_gap
            )
        return merged

    def _calculate_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED: Vectorized feature calculation with robust default Series for
        missing columns, ensuring correct DataFrame structure for validation.
        """
        # --- 1. Initialize and Calculate Spectral Features ---
        # Always initialize energy columns to prevent KeyErrors later
        df['total_energy'] = np.nan
        df['swell_energy'] = np.nan
        df['wind_energy'] = np.nan
        df['swell_fraction'] = np.nan

        spec_cols = [col for col in df.columns if col.startswith('freq_')]
        if spec_cols:
            spectra = df[spec_cols].values
            df['total_energy'] = calculate_spectral_moments(spectra, self.target_freq_centers, moment=0)
            
            wind_speeds = df.get('wave_wind_speed', pd.Series(0, index=df.index)).fillna(0).values
            swell_energy, wind_energy = separate_swell_wind(spectra, self.target_freq_centers, wind_speeds)
            
            df['swell_energy'] = swell_energy
            df['wind_energy'] = wind_energy
            df['swell_fraction'] = safe_divide(swell_energy, df['total_energy'])

        # --- 2. Calculate Derived & Cross-Source Features ---
        # These columns are initialized to NaN implicitly if their source columns don't exist
        if 'r1' in df.columns and 'r2' in df.columns:
            df['bimodality'] = df['r1'] + df['r2'] - 1.0
        
        if 'wave_wind_dir' in df.columns and 'spec_mean_wave_dir' in df.columns:
            diff = np.abs(df['spec_mean_wave_dir'] - df['wave_wind_dir'])
            df['wave_wind_alignment'] = np.minimum(diff, 360 - diff)
        
        if 'spec_significant_wave_height' in df.columns and df['total_energy'].notna().any():
            expected_energy = (df.get('spec_significant_wave_height', 0) ** 2) / 16.0
            df['energy_discrepancy'] = np.abs(df['total_energy'] - expected_energy)

        # --- 3. Construct Final Feature DataFrame ---
        # FIX: Use a default pd.Series for all .get() calls to ensure correct types
        output_features = {
            'significant_wave_height': df.get('spec_significant_wave_height', pd.Series(np.nan, index=df.index)),
            'peak_period': df.get('wave_peak_period', pd.Series(np.nan, index=df.index)),
            'mean_wave_dir': df.get('spec_mean_wave_dir', pd.Series(np.nan, index=df.index)),
            'wind_speed': df.get('wave_wind_speed', pd.Series(np.nan, index=df.index)),
            'wind_dir': df.get('wave_wind_dir', pd.Series(np.nan, index=df.index)),

            # These are guaranteed to exist because we initialized them
            'total_energy': df['total_energy'],
            'swell_energy': df['swell_energy'],
            'wind_energy': df['wind_energy'],
            'swell_fraction': df['swell_fraction'],
            
            # Directional features
            'alpha1': df.get('alpha1', pd.Series(np.nan, index=df.index)),
            'alpha2': df.get('alpha2', pd.Series(np.nan, index=df.index)),
            'r1': df.get('r1', pd.Series(np.nan, index=df.index)),
            'r2': df.get('r2', pd.Series(np.nan, index=df.index)),
            
            # Derived features
            'bimodality': df.get('bimodality', pd.Series(np.nan, index=df.index)),
            'wave_wind_alignment': df.get('wave_wind_alignment', pd.Series(np.nan, index=df.index)),
            'energy_discrepancy': df.get('energy_discrepancy', pd.Series(np.nan, index=df.index)),
        }
        
        final_df = pd.DataFrame(output_features, index=df.index)
        
        # Drop rows where total_energy could not be calculated, as they are essential
        return final_df.dropna(subset=['total_energy'], how='all')