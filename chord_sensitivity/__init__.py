"""
CHORD Sensitivity Calculator

A package to compute sensitivity of CHORD to Radio Recombination Lines.
"""

__version__ = "0.1.0"
__author__ = "Duncan Ststevens"

from .sensitivity import calculate_sensitivity

__all__ = ["__version__", "__author__", "calculate_sensitivity"]
