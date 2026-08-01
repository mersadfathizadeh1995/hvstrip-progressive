"""Standalone entry point for the HV Strip house-style workbench.

Run:  python -m HV_Strip_Progressive.gui.v2.app
(the ``hv_strip.bat`` target; PYTHONPATH must carry BOTH the distribution
root and ``HV_Pro`` for the shared ``theme_core``.)

Primes the process theme from the shared authority bucket (ADR-0014: the
Hub is the sole mode WRITER; standalone windows read the persisted mode,
overridable via ``HV_STRIP_THEME``) so popups match, then the window themes
itself amber per-window.
"""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from hvsr_pro.packages.theme_core import theme_authority

    from HV_Strip_Progressive.gui.v2.main_window import StripMainWindow
    from HV_Strip_Progressive.gui.v2.theme import apply_theme

    mode = theme_authority.load(env_var="HV_STRIP_THEME")
    apply_theme(app, mode)   # process baseline for unparented popups

    window = StripMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
