"""``repolish`` — the single dynamic-property re-style choke-point.

The house style flips a widget's appearance by setting a **dynamic QSS
property** (e.g. ``step="active"``, ``primary="true"``) and then forcing Qt to
re-evaluate the stylesheet for that widget.  Every house widget routes that
"set property + re-polish" through this one helper so re-theming and state
flips behave identically everywhere (no hard-coded hex, no per-widget
``unpolish``/``polish`` copy-paste).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QWidget


def repolish(widget: QWidget, name: str, value: Any) -> None:
    """Set dynamic property *name*=*value* on *widget* and re-apply its QSS.

    Always re-polishes (even when the value is unchanged) so a palette/QSS swap
    re-evaluates the rule for this widget.
    """
    widget.setProperty(name, value)
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


__all__ = ["repolish"]
