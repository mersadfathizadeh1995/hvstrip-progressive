"""Profile Wizard View — step-by-step per-profile peak selection.

Canvas tab for Forward Multiple mode.  Presents each profile's HV curve
one at a time so the user can pick primary + secondary peaks, optionally
view the Vs profile with Vs30/VsAvg, and navigate through all profiles.
"""
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QCheckBox,
    QDoubleSpinBox, QComboBox, QSizePolicy, QFrame,
)

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.style_constants import BUTTON_PRIMARY, BUTTON_SUCCESS, EMOJI


class ProfileWizardView(QWidget):
    """Step-through wizard for selecting peaks on each profile's HV curve."""

    wizard_finished = pyqtSignal(dict)  # {name: {"f0": (f,a,i), "secondary": [...]}}

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._results = []          # list of ProfileResult
        self._peak_data = {}        # {name: {"f0": tuple|None, "secondary": [tuple]}}
        self._ann_positions = {}    # {name: {freq_key: (x, y)}} for drag annotation
        self._current_idx = 0
        self._pick_f0 = False
        self._pick_sec = False
        self._drag_start = None          # (freq, amp, idx) during drag
        self._drag_temp_marker = None    # temporary Line2D
        self._bedrock_profile_name = None  # track which profile bedrock combo is for
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: profile list ───────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)

        left_lay.addWidget(QLabel("Profiles:"))
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_list_row)
        left_lay.addWidget(self._list, 1)

        self._info = QLabel("0 / 0")
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setStyleSheet("font-weight: bold; font-size: 11px;")
        left_lay.addWidget(self._info)

        left.setFixedWidth(180)
        splitter.addWidget(left)

        # ── Right: HV canvas + controls ──────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(2, 2, 2, 2)

        # Top info bar
        top = QHBoxLayout()
        self._title_label = QLabel("No profiles loaded")
        self._title_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        top.addWidget(self._title_label, 1)
        right_lay.addLayout(top)

        # Main split: HV canvas + optional Vs panel
        self._main_split = QSplitter(Qt.Horizontal)

        # HV canvas
        self._hv_plot = MatplotlibWidget(figsize=(10, 5))
        self._hv_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._hv_plot.canvas.mpl_connect("button_press_event", self._on_press)
        self._hv_plot.canvas.mpl_connect("button_release_event", self._on_release)
        self._main_split.addWidget(self._hv_plot)

        # Vs panel (collapsible)
        self._vs_panel = QWidget()
        vs_lay = QVBoxLayout(self._vs_panel)
        vs_lay.setContentsMargins(2, 2, 2, 2)
        self._vs_plot = MatplotlibWidget(figsize=(3, 5))
        self._vs_plot.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        vs_lay.addWidget(self._vs_plot, 1)

        vs_opts = QHBoxLayout()
        self._chk_vs30 = QCheckBox("Vs30")
        self._chk_vs30.setChecked(True)
        self._chk_vs30.toggled.connect(lambda: self._redraw_vs())
        vs_opts.addWidget(self._chk_vs30)
        self._chk_vsavg = QCheckBox("VsAvg")
        self._chk_vsavg.toggled.connect(lambda: self._redraw_vs())
        vs_opts.addWidget(self._chk_vsavg)
        vs_lay.addLayout(vs_opts)

        # Bedrock interface row (own row, no fixed width)
        bedrock_row = QHBoxLayout()
        bedrock_row.addWidget(QLabel("Bedrock:"))
        self._bedrock_combo = QComboBox()
        self._bedrock_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._bedrock_combo.setMinimumContentsLength(14)
        self._bedrock_combo.currentIndexChanged.connect(lambda: self._redraw_vs())
        bedrock_row.addWidget(self._bedrock_combo, 1)
        vs_lay.addLayout(bedrock_row)

        self._vs_label = QLabel("")
        self._vs_label.setStyleSheet("font-size: 9px; color: gray;")
        vs_lay.addWidget(self._vs_label)

        self._main_split.addWidget(self._vs_panel)
        self._main_split.setSizes([700, 250])
        right_lay.addWidget(self._main_split, 1)

        # Peak picking controls
        ctrl = QHBoxLayout()
        self._btn_f0 = QPushButton(f"{EMOJI.get('peak', '🔴')} Select f0")
        self._btn_f0.setCheckable(True)
        self._btn_f0.toggled.connect(self._toggle_f0)
        ctrl.addWidget(self._btn_f0)

        self._btn_sec = QPushButton("Select Secondary")
        self._btn_sec.setCheckable(True)
        self._btn_sec.toggled.connect(self._toggle_sec)
        ctrl.addWidget(self._btn_sec)

        btn_clear = QPushButton("Clear Sec.")
        btn_clear.clicked.connect(self._clear_sec)
        ctrl.addWidget(btn_clear)

        self._chk_show_vs = QCheckBox("Show Vs")
        self._chk_show_vs.setChecked(True)
        self._chk_show_vs.toggled.connect(self._toggle_vs_panel)
        ctrl.addWidget(self._chk_show_vs)

        ctrl.addStretch()
        right_lay.addLayout(ctrl)

        # Selection label
        self._sel_label = QLabel("Click on the curve to select peaks")
        self._sel_label.setStyleSheet("font-size: 10px; color: #555;")
        right_lay.addWidget(self._sel_label)

        # Navigation
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Previous")
        self._btn_prev.clicked.connect(self._go_prev)
        nav.addWidget(self._btn_prev)

        self._btn_next = QPushButton("Next ▶")
        self._btn_next.clicked.connect(self._go_next)
        nav.addWidget(self._btn_next)

        nav.addStretch()

        self._btn_finish = QPushButton(f"{EMOJI.get('run', '✅')} Finish")
        self._btn_finish.setStyleSheet(BUTTON_SUCCESS)
        self._btn_finish.clicked.connect(self._on_finish)
        nav.addWidget(self._btn_finish)
        right_lay.addLayout(nav)

        splitter.addWidget(right)
        splitter.setSizes([180, 800])
        outer.addWidget(splitter)

    # ── Public API ─────────────────────────────────────────────

    def set_results(self, results, auto_peaks=None):
        """Load ProfileResults into wizard.

        Parameters
        ----------
        results : list[ProfileResult]
            Computed profile results.
        auto_peaks : dict or None
            Pre-detected peaks: {name: {"f0": (f,a,i), "secondary": [...]}}
        """
        self._results = [r for r in results if r.computed]
        self._peak_data = {}
        self._bedrock_profile_name = None  # force bedrock repopulation

        self._list.clear()
        for r in self._results:
            pk = (auto_peaks or {}).get(r.name, {})
            self._peak_data[r.name] = {
                "f0": pk.get("f0", r.f0),
                "secondary": list(pk.get("secondary", r.secondary_peaks or [])),
            }
            has_peak = self._peak_data[r.name]["f0"] is not None
            icon = "✓" if has_peak else "  "
            self._list.addItem(f"{icon}  {r.name}")

        if self._results:
            self._current_idx = 0
            self._list.setCurrentRow(0)
            self._show_profile(0)
        self._update_nav()

    def get_peak_data(self):
        return dict(self._peak_data)

    # ── Navigation ─────────────────────────────────────────────

    def _on_list_row(self, row):
        if 0 <= row < len(self._results):
            self._current_idx = row
            self._show_profile(row)
            self._update_nav()

    def _go_prev(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._list.setCurrentRow(self._current_idx)

    def _go_next(self):
        if self._current_idx < len(self._results) - 1:
            self._current_idx += 1
            self._list.setCurrentRow(self._current_idx)

    def _update_nav(self):
        n = len(self._results)
        self._btn_prev.setEnabled(self._current_idx > 0)
        self._btn_next.setEnabled(self._current_idx < n - 1)
        self._info.setText(f"{self._current_idx + 1} / {n}")

    def _on_finish(self):
        self.wizard_finished.emit(self.get_peak_data())

    # ── Display ────────────────────────────────────────────────

    def _show_profile(self, idx):
        if idx < 0 or idx >= len(self._results):
            return
        r = self._results[idx]
        self._title_label.setText(f"Profile: {r.name}")
        self._redraw_hv(r)
        self._redraw_vs(r)

    def _redraw_hv(self, r=None):
        if r is None and self._results:
            r = self._results[self._current_idx]
        if r is None:
            return

        fig = self._hv_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)

        ax.plot(r.freqs, r.amps, color="royalblue", lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("H/V Ratio")
        ax.set_title(r.name, fontsize=10)
        ax.grid(True, alpha=0.3, which="both")

        # Draw peaks from peak_data with drag annotation positions
        pk = self._peak_data.get(r.name, {})
        name_ann = self._ann_positions.get(r.name, {})
        f0 = pk.get("f0")
        if f0:
            ax.plot(f0[0], f0[1], "*", color="red", ms=14, zorder=10,
                    markeredgecolor="darkred", markeredgewidth=0.8)
            ax.axvline(f0[0], color="red", ls="--", lw=0.8, alpha=0.4)
            ann_key = f"f0_{f0[0]:.6f}"
            ann_pos = name_ann.get(ann_key)
            if ann_pos:
                ax.annotate(
                    f"f0 = {f0[0]:.4f} Hz\nA = {f0[1]:.2f}",
                    xy=(f0[0], f0[1]), xycoords="data",
                    xytext=ann_pos, textcoords="data",
                    fontsize=8, color="red", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
            else:
                ylim = ax.get_ylim()
                y_range = ylim[1] - ylim[0]
                yx, yy = (12, -20) if f0[1] > ylim[0] + 0.8 * y_range else (12, 12)
                ax.annotate(
                    f"f0 = {f0[0]:.4f} Hz\nA = {f0[1]:.2f}",
                    xy=(f0[0], f0[1]), xytext=(yx, yy),
                    textcoords="offset points", fontsize=8, color="red",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        sec = pk.get("secondary", [])
        colors_sec = ["green", "purple", "orange", "brown", "teal"]
        for j, s in enumerate(sec):
            sc = colors_sec[j % len(colors_sec)]
            ax.plot(s[0], s[1], "*", color=sc, ms=11, zorder=9,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.axvline(s[0], color=sc, ls=":", lw=0.7, alpha=0.4)
            ann_key = f"sec_{j}_{s[0]:.6f}"
            ann_pos = name_ann.get(ann_key)
            if ann_pos:
                ax.annotate(
                    f"Sec.{j+1}: {s[0]:.3f} Hz\nA = {s[1]:.2f}",
                    xy=(s[0], s[1]), xycoords="data",
                    xytext=ann_pos, textcoords="data",
                    fontsize=7, color=sc,
                    arrowprops=dict(arrowstyle="->", color=sc, lw=0.6),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#f0f0f0", ec=sc, alpha=0.8))
            else:
                y_off = 10 if j % 2 == 0 else -18
                ax.annotate(
                    f"Sec.{j+1}: {s[0]:.3f} Hz\nA = {s[1]:.2f}",
                    xy=(s[0], s[1]), xytext=(8, y_off),
                    textcoords="offset points", fontsize=7, color=sc,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#f0f0f0", ec=sc, alpha=0.8))

        fig.tight_layout()
        self._hv_plot.refresh()
        self._update_sel_label()

    def _redraw_vs(self, r=None):
        if r is None and self._results:
            r = self._results[self._current_idx]
        if r is None or not r.profile:
            return
        if not self._chk_show_vs.isChecked():
            return

        prof = r.profile

        # Only re-populate bedrock combo when showing a DIFFERENT profile
        if self._bedrock_profile_name != r.name:
            self._bedrock_profile_name = r.name
            self._populate_bedrock_combo(prof)

        fig = self._vs_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)

        depths, vs = [], []
        z = 0.0
        finite = [L for L in prof.layers if not L.is_halfspace]
        hs = [L for L in prof.layers if L.is_halfspace]
        for L in finite:
            depths.append(z); vs.append(L.vs)
            z += L.thickness
            depths.append(z); vs.append(L.vs)
        total = z
        if hs:
            hd = total * 0.25
            depths.append(z); vs.append(hs[0].vs)
            z += hd
            depths.append(z); vs.append(hs[0].vs)

        # Plot data first (prevents singular-matrix on axhline)
        ax.plot(vs, depths, color="teal", lw=1.8)
        ax.fill_betweenx(depths, 0, vs, alpha=0.1, color="teal")
        ax.axhline(total, color="red", lw=0.8, ls="--", alpha=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=8)
        ax.set_ylabel("Depth (m)", fontsize=8)
        ax.set_title("Vs Profile", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

        vs_max = max(vs) if vs else 500

        # Force axis limits so transforms are resolved before axhline
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        fig.canvas.draw()

        info_parts = []

        # Vs30
        if self._chk_vs30.isChecked():
            try:
                from hvstrip_progressive.core.vs_average import vs_average_from_profile
                res = vs_average_from_profile(prof, target_depth=30.0)
                ax.axhline(30.0, color="blue", lw=0.8, ls="-.", alpha=0.6)
                ax.annotate(f"Vs30 = {res.vs_avg:.0f} m/s",
                            xy=(vs_max * 0.5, 30.0),
                            xytext=(0, -10), textcoords="offset points",
                            fontsize=7, color="blue", fontweight="bold")
                info_parts.append(f"Vs30 = {res.vs_avg:.0f} m/s")
            except Exception:
                pass

        # VsAvg to bedrock
        if self._chk_vsavg.isChecked() and self._bedrock_combo.count() > 0:
            try:
                from hvstrip_progressive.core.vs_average import compute_vs_average
                bd_idx = self._bedrock_combo.currentData()
                if bd_idx is not None:
                    finite_local = [L for L in prof.layers if not L.is_halfspace]
                    if bd_idx < len(finite_local):
                        bd = sum(L.thickness for L in finite_local[:bd_idx + 1])
                        layers = [(L.thickness, L.vs) for L in finite_local]
                        res = compute_vs_average(layers, target_depth=bd, use_halfspace=False)
                        ax.axhline(bd, color="darkred", lw=1.2, ls="--", alpha=0.7)
                        ax.annotate(f"VsAvg = {res.vs_avg:.0f} m/s",
                                    xy=(vs_max * 0.3, bd),
                                    xytext=(0, 6), textcoords="offset points",
                                    fontsize=7, color="darkred", fontweight="bold")
                        info_parts.append(f"VsAvg = {res.vs_avg:.0f} m/s (to {bd:.1f}m)")
            except Exception:
                pass

        # Update info label below Vs plot
        self._vs_label.setText("  |  ".join(info_parts) if info_parts else "")

        fig.tight_layout()
        self._vs_plot.refresh()

    def _populate_bedrock_combo(self, profile):
        self._bedrock_combo.blockSignals(True)
        self._bedrock_combo.clear()
        finite = [L for L in profile.layers if not L.is_halfspace]
        z = 0.0
        for i, L in enumerate(finite):
            z += L.thickness
            self._bedrock_combo.addItem(f"Interface {i+1} ({z:.1f}m)", i)
        if finite:
            self._bedrock_combo.setCurrentIndex(len(finite) - 1)
        self._bedrock_combo.blockSignals(False)

    # ── Peak picking ───────────────────────────────────────────

    def _toggle_f0(self, on):
        self._pick_f0 = on
        if on:
            self._pick_sec = False
            self._btn_sec.setChecked(False)
        self._btn_f0.setStyleSheet(
            "background-color: #90EE90;" if on else "")

    def _toggle_sec(self, on):
        self._pick_sec = on
        if on:
            self._pick_f0 = False
            self._btn_f0.setChecked(False)
        self._btn_sec.setStyleSheet(
            "background-color: #FFD580;" if on else "")

    def _clear_sec(self):
        if self._results:
            name = self._results[self._current_idx].name
            self._peak_data.setdefault(name, {})["secondary"] = []
            # Clear annotation positions for secondary peaks
            if name in self._ann_positions:
                keys_to_del = [k for k in self._ann_positions[name] if k.startswith("sec_")]
                for k in keys_to_del:
                    del self._ann_positions[name][k]
            self._redraw_hv()

    def _toggle_vs_panel(self, show):
        self._vs_panel.setVisible(show)
        if show:
            self._redraw_vs()

    def _on_press(self, event):
        """Mouse press: snap to HV curve, show temporary marker."""
        if event.inaxes is None:
            return
        if not self._results:
            return

        # Right-click: remove nearest peak
        if event.button == 3:
            self._remove_nearest_peak(event.xdata)
            return

        if event.button != 1:
            return
        if not (self._pick_f0 or self._pick_sec):
            return

        r = self._results[self._current_idx]
        cx = event.xdata
        if cx is None or r.freqs is None:
            return

        amp = float(np.interp(cx, r.freqs, r.amps))
        nearest_idx = int(np.argmin(np.abs(r.freqs - cx)))
        self._drag_start = (float(r.freqs[nearest_idx]), float(r.amps[nearest_idx]), nearest_idx)

        color = "red" if self._pick_f0 else "green"
        ax = self._hv_plot.figure.axes[0] if self._hv_plot.figure.axes else None
        if ax:
            self._drag_temp_marker, = ax.plot(
                self._drag_start[0], self._drag_start[1], "*",
                color=color, ms=14, markeredgecolor="black",
                markeredgewidth=0.8, zorder=20)
            self._hv_plot.canvas.draw_idle()

    def _on_release(self, event):
        """Mouse release: place annotation at release point with arrow to peak."""
        if self._drag_start is None or event.button != 1:
            return

        freq, amp, nearest_idx = self._drag_start
        self._drag_start = None

        # Remove temp marker
        if self._drag_temp_marker is not None:
            try:
                self._drag_temp_marker.remove()
            except Exception:
                pass
            self._drag_temp_marker = None

        r = self._results[self._current_idx]
        name = r.name
        peak = (freq, amp, nearest_idx)
        self._peak_data.setdefault(name, {"f0": None, "secondary": []})

        # Store annotation position from release point
        self._ann_positions.setdefault(name, {})
        if self._pick_f0:
            self._peak_data[name]["f0"] = peak
            ann_key = f"f0_{freq:.6f}"
            if event.inaxes is not None and event.xdata is not None:
                self._ann_positions[name][ann_key] = (event.xdata, event.ydata)
        elif self._pick_sec:
            self._peak_data[name]["secondary"].append(peak)
            j = len(self._peak_data[name]["secondary"]) - 1
            ann_key = f"sec_{j}_{freq:.6f}"
            if event.inaxes is not None and event.xdata is not None:
                self._ann_positions[name][ann_key] = (event.xdata, event.ydata)

        # Update list icon
        has_f0 = self._peak_data[name]["f0"] is not None
        icon = "✓" if has_f0 else "  "
        item = self._list.item(self._current_idx)
        if item:
            item.setText(f"{icon}  {r.name}")

        self._redraw_hv(r)

    def _remove_nearest_peak(self, xdata):
        """Remove the nearest peak to the clicked x position (right-click)."""
        if xdata is None or not self._results:
            return
        r = self._results[self._current_idx]
        name = r.name
        pk = self._peak_data.get(name, {})

        all_peaks = []
        f0 = pk.get("f0")
        if f0:
            all_peaks.append(("f0", 0, f0))
        for i, sp in enumerate(pk.get("secondary", [])):
            all_peaks.append(("sec", i, sp))
        if not all_peaks:
            return

        dists = [abs(xdata - p[2][0]) for p in all_peaks]
        nearest = all_peaks[int(np.argmin(dists))]

        if nearest[0] == "f0":
            self._peak_data[name]["f0"] = None
            if name in self._ann_positions:
                keys_del = [k for k in self._ann_positions[name] if k.startswith("f0_")]
                for k in keys_del:
                    del self._ann_positions[name][k]
        else:
            idx = nearest[1]
            self._peak_data[name]["secondary"].pop(idx)
            if name in self._ann_positions:
                keys_del = [k for k in self._ann_positions[name] if k.startswith(f"sec_{idx}_")]
                for k in keys_del:
                    del self._ann_positions[name][k]

        # Update list icon
        has_f0 = self._peak_data.get(name, {}).get("f0") is not None
        icon = "✓" if has_f0 else "  "
        item = self._list.item(self._current_idx)
        if item:
            item.setText(f"{icon}  {r.name}")

        self._redraw_hv(r)

    def _update_sel_label(self):
        if not self._results:
            self._sel_label.setText("No profiles loaded")
            return
        r = self._results[self._current_idx]
        pk = self._peak_data.get(r.name, {})
        parts = []
        f0 = pk.get("f0")
        if f0:
            parts.append(f"f0 = {f0[0]:.4f} Hz (A = {f0[1]:.2f})")
        sec = pk.get("secondary", [])
        for j, s in enumerate(sec):
            parts.append(f"Sec.{j+1} = {s[0]:.3f} Hz")
        self._sel_label.setText(" | ".join(parts) if parts else "No peaks selected")
