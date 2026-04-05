"""Summary Dock Panel — collapsible right-side dock for Forward Multiple.

Shows a compact table of all profiles with f0, amplitude, secondary peaks,
Vs30, VsAvg, and computation status.  Designed to be tabified with the Log
dock on the right side of the main window.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QAbstractItemView,
)


class SummaryDockWidget(QDockWidget):
    """Collapsible right-side dock that shows profile summary table."""

    COLUMNS = ["Profile", "f0 (Hz)", "Amp", "Sec.Peaks", "Vs30", "VsAvg", "Status"]
    COL_WIDTHS = [90, 60, 50, 80, 55, 55, 55]

    def __init__(self, parent=None):
        super().__init__("Summary", parent)
        self.setObjectName("SummaryDock")
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        self._container = QWidget()
        lay = QVBoxLayout(self._container)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        lbl = QLabel("<b>Profile Summary</b>")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { font-size: 11px; }"
            "QHeaderView::section { font-size: 10px; padding: 2px; }"
        )

        hdr = self._table.horizontalHeader()
        for i, w in enumerate(self.COL_WIDTHS):
            self._table.setColumnWidth(i, w)
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(QHeaderView.Interactive)

        lay.addWidget(self._table, 1)
        self.setWidget(self._container)
        self.setMinimumWidth(200)

    # ── Public API ─────────────────────────────────────────────

    def set_results(self, results, peak_data=None):
        """Populate table from list of ProfileResult + optional peak_data dict."""
        self._table.setRowCount(0)
        peak_data = peak_data or {}

        for r in results:
            row = self._table.rowCount()
            self._table.insertRow(row)

            pk = peak_data.get(r.name, {})
            f0 = pk.get("f0") or getattr(r, "f0", None)

            self._table.setItem(row, 0, QTableWidgetItem(r.name))
            self._table.setItem(row, 1, QTableWidgetItem(
                f"{f0[0]:.3f}" if f0 else "—"))
            self._table.setItem(row, 2, QTableWidgetItem(
                f"{f0[1]:.2f}" if f0 else "—"))

            sec = pk.get("secondary", getattr(r, "secondary_peaks", None) or [])
            sec_str = ", ".join(f"{s[0]:.2f}" for s in sec) if sec else "—"
            self._table.setItem(row, 3, QTableWidgetItem(sec_str))

            # Vs30 / VsAvg
            vs30 = "—"
            vsavg = "—"
            if r.profile:
                # Vs30
                try:
                    from ...core.vs_average import vs_average_from_profile
                    res30 = vs_average_from_profile(r.profile, target_depth=30.0)
                    vs30 = f"{res30.vs_avg:.0f}"
                except Exception:
                    pass
                # VsAvg: try deepest finite interface as bedrock
                try:
                    from ...core.vs_average import compute_vs_average
                    finite = [L for L in r.profile.layers if not L.is_halfspace]
                    if finite:
                        bd = sum(L.thickness for L in finite)
                        layers = [(L.thickness, L.vs) for L in finite]
                        if bd > 0 and layers:
                            res_avg = compute_vs_average(layers, bd, use_halfspace=False)
                            vsavg = f"{res_avg.vs_avg:.0f}"
                except Exception:
                    pass
                # Fallback: use vs_average_from_profile with total depth
                if vsavg == "—":
                    try:
                        from ...core.vs_average import vs_average_from_profile
                        finite = [L for L in r.profile.layers if not L.is_halfspace]
                        total = sum(L.thickness for L in finite)
                        if total > 0:
                            res_fb = vs_average_from_profile(r.profile, target_depth=total)
                            vsavg = f"{res_fb.vs_avg:.0f}"
                    except Exception:
                        pass
            self._table.setItem(row, 4, QTableWidgetItem(vs30))
            self._table.setItem(row, 5, QTableWidgetItem(vsavg))
            self._table.setItem(row, 6, QTableWidgetItem(
                "✓" if r.computed else "—"))

    def update_profile(self, idx, result, peak_info=None):
        """Update a single row when a profile is computed or peaks change."""
        if idx >= self._table.rowCount():
            return
        pk = peak_info or {}
        f0 = pk.get("f0") or getattr(result, "f0", None)

        self._table.setItem(idx, 0, QTableWidgetItem(result.name))
        self._table.setItem(idx, 1, QTableWidgetItem(
            f"{f0[0]:.3f}" if f0 else "—"))
        self._table.setItem(idx, 2, QTableWidgetItem(
            f"{f0[1]:.2f}" if f0 else "—"))

        sec = pk.get("secondary",
                      getattr(result, "secondary_peaks", None) or [])
        sec_str = ", ".join(f"{s[0]:.2f}" for s in sec) if sec else "—"
        self._table.setItem(idx, 3, QTableWidgetItem(sec_str))

        self._table.setItem(idx, 6, QTableWidgetItem(
            "✓" if result.computed else "—"))

    def clear(self):
        self._table.setRowCount(0)
