"""
Input validation utilities.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def validate_model_file(model_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate HVf-format velocity model file.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        if not model_path.exists():
            return False, f"Model file not found: {model_path}"
        
        with open(model_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return False, "Empty model file"
        
        # Check first line (number of layers)
        try:
            n_layers = int(lines[0].strip())
        except ValueError:
            return False, "First line must be integer (number of layers)"
        
        if n_layers <= 0:
            return False, "Number of layers must be positive"
        
        if len(lines) < n_layers + 1:
            return False, f"Expected {n_layers} layer rows, found {len(lines)-1}"
        
        # Validate layer data
        for i in range(1, n_layers + 1):
            parts = lines[i].strip().split()
            if len(parts) < 4:
                return False, f"Layer {i} has fewer than 4 parameters"
            
            try:
                thickness, vp, vs, rho = map(float, parts[:4])
            except ValueError:
                return False, f"Layer {i} contains non-numeric values"
            
            if thickness < 0:
                return False, f"Layer {i} has negative thickness"
            if vp <= 0 or vs <= 0 or rho <= 0:
                return False, f"Layer {i} has non-positive velocity or density"
        
        # Check half-space (last layer should have thickness = 0)
        last_parts = lines[n_layers].strip().split()
        if float(last_parts[0]) != 0.0:
            return False, "Last layer must be half-space (thickness = 0)"
        
        return True, None
        
    except Exception as e:
        return False, f"Error reading model file: {e}"


def validate_hv_csv(csv_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate HVSR curve CSV file.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        if not csv_path.exists():
            return False, f"CSV file not found: {csv_path}"
        
        # Try to read as numpy array
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        
        if data.ndim != 2 or data.shape[1] < 2:
            return False, "CSV must have at least 2 columns (frequency, amplitude)"
        
        freqs = data[:, 0]
        amps = data[:, 1]
        
        if len(freqs) < 3:
            return False, "Need at least 3 frequency points"
        
        if np.any(freqs <= 0):
            return False, "Frequencies must be positive"
        
        if np.any(amps < 0):
            return False, "Amplitudes cannot be negative"
        
        if not np.all(np.diff(freqs) > 0):
            return False, "Frequencies must be increasing"
        
        return True, None
        
    except Exception as e:
        return False, f"Error reading CSV file: {e}"


__all__ = [
    "validate_model_file",
    "validate_hv_csv"
]
