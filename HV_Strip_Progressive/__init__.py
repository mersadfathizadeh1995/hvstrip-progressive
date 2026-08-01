"""HV Strip Progressive — progressive HVSR layer stripping.

Core-API-Consumers package: ``core/`` (pure compute + engines), ``api/``
(the ``HVStripAnalysis`` facade), ``gui/`` (the desktop app), ``research/``.

The package import stays **Qt-free** (the layering guard enforces it): the
GUI window is exported LAZILY via ``__getattr__`` so headless consumers
(api scripts, research/, tests, a future CLI/MCP) never pay the Qt cost —
``from HV_Strip_Progressive import HVStripWindow`` still works unchanged.
"""

from typing import Any

__all__ = ["HVStripWindow"]
__version__ = "2.2.0"


def __getattr__(name: str) -> Any:
    if name == "HVStripWindow":
        from .gui.strip_window import HVStripWindow

        return HVStripWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
