"""HV Strip Progressive — GUI package.

Two trees during the house-style uplift:

* the LEGACY PyQt5 app (``strip_window`` + panels/views/widgets/workers) —
  retiring at the cutover;
* ``gui.v2`` — the PySide6 house-style workbench.

``HVStripWindow`` is exported **lazily** so importing ``gui.v2`` (PySide6)
never executes the PyQt5 legacy modules — the two bindings cannot coexist in
one process.
"""

from typing import Any

__all__ = ["HVStripWindow"]


def __getattr__(name: str) -> Any:
    if name == "HVStripWindow":
        from .strip_window import HVStripWindow

        return HVStripWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
