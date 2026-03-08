"""HV Strip Progressive — PyQt5 GUI for HV_Pro integration."""
import os, sys

# Ensure core modules are importable
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_core_root = os.path.join(_pkg_root, "hvstrip_progressive")
if _core_root not in sys.path:
    sys.path.insert(0, _core_root)

from .strip_window import HVStripWindow

__all__ = ["HVStripWindow"]
__version__ = "2.0.0"
