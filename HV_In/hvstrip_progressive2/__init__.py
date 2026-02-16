"""
HVSR Progressive Layer Stripping Analysis Package
================================================

A comprehensive toolkit for progressive layer stripping analysis of HVSR data.

This package provides:
- Layer stripping algorithms
- HV forward modeling with HVf.exe
- Publication-ready visualization
- Comprehensive analysis reports
- Command-line interface

Example usage:
    >>> from hvstrip_progressive.core import batch_workflow
    >>> results = batch_workflow.run_complete_workflow("model.txt", "output/")
"""

__version__ = "1.0.0"
__author__ = "HVSR-Diffuse Development Team"
__email__ = "your.email@example.com"

# Import main modules for easy access
from .core import (
    stripper,
    hv_forward, 
    hv_postprocess,
    batch_workflow,
    report_generator
)

__all__ = [
    "stripper",
    "hv_forward", 
    "hv_postprocess",
    "batch_workflow",
    "report_generator",
    "__version__",
    "__author__",
    "__email__"
]
