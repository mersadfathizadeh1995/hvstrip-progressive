"""Standalone entry point for the LEGACY PyQt5 HV Strip window.

TEMPORARY (retired with the legacy GUI at the house-style cutover): the old
``HVStripWindow`` was only ever constructed by the legacy HV Pro app's
submodule manager and has no ``main()`` of its own.  This launcher exists so
the old GUI can be run side-by-side with the new house-style workbench for
comparison during the rebuild (``HV_Analyze_Pro/hv_strip_old.bat``).

Run:  python -m HV_Strip_Progressive.gui.legacy_main
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("QT_API", "pyqt5")
    import matplotlib

    try:
        matplotlib.use("Qt5Agg")
    except Exception:  # noqa: BLE001 — backend already set
        pass

    from PyQt5.QtWidgets import QApplication

    from HV_Strip_Progressive.gui.strip_window import HVStripWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = HVStripWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
