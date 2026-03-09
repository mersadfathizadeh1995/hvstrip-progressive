"""HV Strip Progressive — PyQt5 GUI for HV_Pro integration.

Core engine modules are internalised under HV_Strip_Progressive.core/
so no sys.path manipulation is needed.
"""

from .strip_window import HVStripWindow

__all__ = ["HVStripWindow"]
__version__ = "2.1.0"
