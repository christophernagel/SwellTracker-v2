import numpy as np
import torch

def calculate_spectral_moments(spectra, frequencies, moment=0):
    """
    Vectorized spectral moment calculation.
    
    Args:
        spectra (np.array): 2D array of spectra (n_spectra, n_frequencies).
        frequencies (np.array): 1D array of frequency values.
        moment (int): The moment to calculate.
        
    Returns:
        np.array: 1D array of the calculated moments.
    """
    return np.trapz(spectra * (frequencies[np.newaxis, :] ** moment), x=frequencies, axis=1)

def separate_swell_wind(spectra, frequencies, wind_speeds):
    """
    Vectorized swell/wind energy separation using broadcasting.
    
    Args:
        spectra (np.array): 2D array of spectra (n_spectra, n_frequencies).
        frequencies (np.array): 1D array of frequency values.
        wind_speeds (np.array): 1D array of wind speeds.
        
    Returns:
        tuple[np.array, np.array]: A tuple containing swell energies and wind energies.
    """
    # Create a 1D array of cutoff frequencies, one for each spectrum
    swell_cutoffs = np.where(wind_speeds > 5, 0.1 * wind_speeds, 0.15)
    
    # Use broadcasting to create a 2D boolean mask
    swell_mask = frequencies[np.newaxis, :] < swell_cutoffs[:, np.newaxis]
    wind_mask = ~swell_mask
    
    # Use the mask to zero out irrelevant parts and integrate in one pass
    swell_energy = np.trapz(np.where(swell_mask, spectra, 0), x=frequencies, axis=1)
    wind_energy = np.trapz(np.where(wind_mask, spectra, 0), x=frequencies, axis=1)
    
    return swell_energy, wind_energy