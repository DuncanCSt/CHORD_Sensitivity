"""Helper functions for RFI analysis."""

import tempfile
import zipfile
from pathlib import Path

import numpy as np


def load_demo_data():
    """
    Load demo data from data_slice.zip as a memory-mapped array.

    Unzips data_slice.csv and converts to a memory-mapped binary format
    to simulate handling of large real-world data without loading into memory.

    Returns
    -------
    np.memmap
        Memory-mapped array of the demo data.
    """
    zip_path = Path(__file__).parent.parent.parent / "rfi_data" / "data_slice.zip"

    if not zip_path.exists():
        raise FileNotFoundError(f"data_slice.zip not found at {zip_path}")

    # Create a temporary directory for extracted files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract the CSV
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)

        # Load CSV as array
        csv_path = tmpdir / "data_slice.csv"
        data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)

        # Save as binary file in a persistent temp location for memmap
        memmap_path = Path(tempfile.gettempdir()) / "data_slice.dat"
        data.astype(np.float32).tofile(memmap_path)

    # Return as memory-mapped array
    memmap = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=data.shape)
    return memmap
