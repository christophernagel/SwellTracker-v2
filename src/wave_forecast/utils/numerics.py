import numpy as np

def safe_divide(numerator, denominator, default_val=0.0):
    """
    Vectorized safe division to avoid divide-by-zero errors.
    
    Args:
        numerator (np.array): The numerator.
        denominator (np.array): The denominator.
        default_val (float): Value to return for invalid divisions.
    
    Returns:
        np.array: The result of the division.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
        result[~np.isfinite(result)] = default_val
    return result