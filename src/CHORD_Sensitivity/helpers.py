from __future__ import annotations

import numpy as np

def convert_freq_to_Hz(freq, params) -> float | list[float]:
    """
    Automatically detect range and convert to Hz.
    
    Parameters
    ----------
    freq : float or array-like
        Frequency value(s) in Hz, MHz, or GHz.
    
    Returns
    -------
    float or list[float]
        Frequency value(s) in Hz.
    """
    
    min_freq_MHz = params['frequencyMin']
    max_freq_MHz = params['frequencyMax']
    
    # Check if input is array-like
    is_array = isinstance(freq, (list, tuple, np.ndarray))
    freq_array = np.asarray(freq) if is_array else np.array([freq])
    
    # Initialize result array
    result = np.zeros_like(freq_array, dtype=float)
    
    for i, f in enumerate(freq_array):
        # Check if frequency is in Hz
        if min_freq_MHz * 1e6 <= f <= max_freq_MHz * 1e6:
            result[i] = f
        # Check if frequency is in MHz
        elif min_freq_MHz <= f <= max_freq_MHz:
            result[i] = f * 1e6
        # Check if frequency is in GHz
        elif (min_freq_MHz / 1e3) <= f <= (max_freq_MHz / 1e3):
            result[i] = f * 1e9
        else:
            raise ValueError(f"Frequency {f} is out of the telescope's operating range. Or not one of the supported units (Hz, MHz, GHz).")
    
    return result.tolist() if is_array else float(result[0])
