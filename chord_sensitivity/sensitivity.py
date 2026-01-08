"""
Example module for CHORD sensitivity calculations.

This module provides basic functionality for computing radio sensitivity.
"""

import numpy as np
from typing import Optional


def calculate_sensitivity(
    system_temperature: float,
    bandwidth: float,
    integration_time: float,
    n_antennas: Optional[int] = 1
) -> float:
    """
    Calculate the sensitivity of a radio telescope.
    
    Parameters
    ----------
    system_temperature : float
        System temperature in Kelvin
    bandwidth : float
        Bandwidth in Hz
    integration_time : float
        Integration time in seconds
    n_antennas : int, optional
        Number of antennas (default is 1)
    
    Returns
    -------
    float
        Sensitivity in Kelvin
    
    Examples
    --------
    >>> calculate_sensitivity(50, 1e6, 3600, 512)
    0.00023148148148148147
    """
    if system_temperature <= 0:
        raise ValueError("System temperature must be positive")
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")
    if integration_time <= 0:
        raise ValueError("Integration time must be positive")
    if n_antennas < 1:
        raise ValueError("Number of antennas must be at least 1")
    
    # Radiometer equation
    sensitivity = system_temperature / np.sqrt(bandwidth * integration_time * n_antennas)
    
    return sensitivity
