"""HV Strip Wizard View — step-through per-step peak selection for stripping.

Canvas tab for Strip Single mode.  Presents each stripping step's HV curve
one at a time so the user can pick primary + secondary peaks, choose bedrock
interface, view Vs30/VsAvg, and navigate through all steps.

Supports two modes:
  - **Automatic**: peaks pre-detected, user can review and click Finish
  - **Interactive**: user steps through each step, selecting peaks manually
"""
import re
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QCheckBox,
    QComboBox, QSizePolicy,
)

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.style_constants import BUTTON_PRIMARY, BUTTON_SUCCESS, EMOJI

SEC_COLORS = ["green", "purple", "orange", "brown", "teal"]


def _natural_sort_key(s):
    """Sort key that orders 'Step0', 'Step1', ..., 'Step10' numerically."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', s)]


class StripWizardView(QWidget):
    """Step-through wizard for selecting peaks on each stripping step's HV curve."""

    wizard_finished = pyqtSignal(dict)
    # Emits: {"step_name": {"f0": (f,a,i), "secondary": [...],
    #                        "bedrock_depth": float, "vs30": float, "vsavg": float}, ...}

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._steps = []            # list of step dicts from workflow result
        self._step_names = []       # ordered step names
        self._peak_data = {}        # {step_name: {"f0": tuple|None, "secondary": [tuple]}}
        self._bedrock_data = {}     # {step_name: {"depth": float, "vs30": float, "vsavg": float}}
        self._current_idx = 0
        self._pick_f0 = False
        self._pick_sec = False
        self._auto_mode = True
        self._auto_peak_cfg = None  # config from AutoPeakSettingsDialog
        self._drag_start_x = None   # for click-and-release peak selection
        self._drag_rect = None      # matplotlib patch for drag visual
        self._build_ui()

    # ══════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: step list + options ────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(4)

        left_lay.addWidget(QLabel("Stripping Steps:"))
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_list_row)
        left_lay.addWidget(self._list, 1)

        self._info = QLabel("0 / 0")
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setStyleSheet("font-weight: bold; font-size: 11px;")
        left_lay.addWidget(self._info)

        left.setFixedWidth(200)
        splitter.addWidget(left)

        # ── Right: HV canvas + Vs panel + controls ───────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(2, 2, 2, 2)
        right_lay.setSpacing(2)

        # Title bar
        top = QHBoxLayout()
        self._title_label = QLabel("No stripping results loaded")
        self._title_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        top.addWidget(self._title_label, 1)

        self._step_info = QLabel("")
        self._step_info.setStyleSheet("font-size: 10px; color: #555;")
        top.addWidget(self._step_info)
        right_lay.addLayout(top)

        # Main split: HV canvas + Vs mini panel
        self._main_split = QSplitter(Qt.Horizontal)

        # HV canvas
        self._hv_plot = MatplotlibWidget(figsize=(10, 5))
        self._hv_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._hv_plot.canvas.mpl_connect("button_press_event", self._on_press)
        self._hv_plot.canvas.mpl_connect("button_release_event", self._on_release)
        self._hv_plot.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._main_split.addWidget(self._hv_plot)

        # Vs mini panel
        self._vs_panel = QWidget()
        vs_lay = QVBoxLayout(self._vs_panel)
        vs_lay.setContentsMargins(2, 2, 2, 2)
        vs_lay.setSpacing(2)
        self._vs_plot = MatplotlibWidget(figsize=(3, 5))
        self._vs_plot.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        vs_lay.addWidget(self._vs_plot, 1)

        vs_opts = QHBoxLayout()
        self._chk_vs30 = QCheckBox("Vs30")
        self._chk_vs30.setChecked(True)
        self._chk_vs30.toggled.connect(self._redraw_vs)
        vs_opts.addWidget(self._chk_vs30)
        self._chk_vsavg = QCheckBox("VsAvg")
        self._chk_vsavg.setChecked(True)
        self._chk_vsavg.toggled.connect(self._redraw_vs)
        vs_opts.addWidget(self._chk_vsavg)
        vs_lay.addLayout(vs_opts)

        # Bedrock interface
        bedrock_row = QHBoxLayout()
        bedrock_row.addWidget(QLabel("Bedrock:"))
        self._bedrock_combo = QComboBox()
        self._bedrock_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._bedrock_combo.setMinimumContentsLength(14)
        self._bedrock_combo.currentIndexChanged.connect(self._on_bedrock_changed)
        bedrock_row.addWidget(self._bedrock_combo, 1)
        vs_lay.addLayout(bedrock_row)

        self._vs_label = QLabel("")
        self._vs_label.setStyleSheet("font-size: 9px; color: gray;")
        self._vs_label.setWordWrap(True)
        vs_lay.addWidget(self._vs_label)

        self._main_split.addWidget(self._vs_panel)
        self._main_split.setSizes([700, 250])
        right_lay.addWidget(self._main_split, 1)

        # Peak picking controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)
        self._btn_f0 = QPushButton(f"{EMOJI.get('peak', '🔴')} Select f0")
        self._btn_f0.setCheckable(True)
        self._btn_f0.setStyleSheet(
            "QPushButton:checked { background-color: #4CAF50; color: white; "
            "font-weight: bold; border-radius: 3px; padding: 3px 8px; }")
        self._btn_f0.toggled.connect(self._toggle_f0)
        ctrl.addWidget(self._btn_f0)

        self._btn_sec = QPushButton("🔶 Select Secondary")
        self._btn_sec.setCheckable(True)
        self._btn_sec.setStyleSheet(
            "QPushButton:checked { background-color: #FF9800; color: white; "
            "font-weight: bold; border-radius: 3px; padding: 3px 8px; }")
        self._btn_sec.toggled.connect(self._toggle_sec)
        ctrl.addWidget(self._btn_sec)

        btn_clear = QPushButton("✕ Clear Peaks")
        btn_clear.clicked.connect(self._clear_peaks)
        ctrl.addWidget(btn_clear)

        self._chk_show_vs = QCheckBox("Show Vs")
        self._chk_show_vs.setChecked(True)
        self._chk_show_vs.toggled.connect(self._toggle_vs_panel)
        ctrl.addWidget(self._chk_show_vs)

        ctrl.addStretch()
        right_lay.addLayout(ctrl)

        # Selection label
        self._sel_label = QLabel("Run stripping to populate wizard")
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
        splitter.setSizes([200, 800])
        outer.addWidget(splitter)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════
    def set_strip_results(self, result_dict):
        """Load stripping workflow results into wizard.

        Parameters
        ----------
        result_dict : dict
            The result from run_complete_workflow() containing
            step_results, strip_directory, etc.
        """
        step_results = result_dict.get("step_results", {})
        if not step_results:
            return

        self._result_dict = result_dict
        self._step_names = sorted(step_results.keys(), key=_natural_sort_key)
        self._steps = []
        self._peak_data = {}
        self._bedrock_data = {}

        for name in self._step_names:
            data = step_results[name]
            # Load HV curve data
            freqs, amps = None, None
            hv_csv = data.get("hv_csv")
            if hv_csv:
                try:
                    arr = np.loadtxt(str(hv_csv), delimiter=",", skiprows=1)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        freqs = arr[:, 0]
                        amps = arr[:, 1]
                except Exception:
                    pass

            # Load model layers
            layers = self._load_model_layers(data.get("model_file"))

            step = {
                "name": name,
                "freqs": freqs,
                "amps": amps,
                "layers": layers,
                "vs30": data.get("vs30"),
                "peak_frequency": data.get("peak_frequency"),
                "peak_amplitude": data.get("peak_amplitude"),
            }
            self._steps.append(step)

            # Auto-detect peak from workflow result
            pf = data.get("peak_frequency")
            pa = data.get("peak_amplitude")
            if pf and pa and freqs is not None:
                idx = int(np.argmin(np.abs(freqs - pf)))
                f0 = (float(pf), float(pa), idx)
            else:
                f0 = None

            self._peak_data[name] = {
                "f0": f0,
                "secondary": [],
            }

            # Default bedrock: deepest finite layer interface
            if layers:
                finite = [L for L in layers if not L.get("is_halfspace", False)]
                total_d = sum(L.get("thickness", 0) for L in finite)
                self._bedrock_data[name] = {
                    "depth": total_d,
                    "vs30": data.get("vs30"),
                    "vsavg": None,
                }
            else:
                self._bedrock_data[name] = {"depth": 0, "vs30": None, "vsavg": None}

        # Auto-detect secondary peaks if config is set
        if self._auto_peak_cfg:
            self._run_auto_detection()

        # Populate list
        self._list.clear()
        for name in self._step_names:
            pk = self._peak_data.get(name, {})
            has_f0 = pk.get("f0") is not None
            # Parse layer count from step name
            n_layers = self._parse_n_layers(name)
            icon = "✓" if has_f0 else "  "
            label = f"{icon}  {name}" if not n_layers else f"{icon}  {name}"
            self._list.addItem(label)

        if self._steps:
            self._current_idx = 0
            self._list.setCurrentRow(0)
            self._show_step(0)
        self._update_nav()

    def get_peak_data(self):
        """Return peak + bedrock data for all steps."""
        result = {}
        for name in self._step_names:
            pk = self._peak_data.get(name, {})
            bd = self._bedrock_data.get(name, {})
            result[name] = {
                "f0": pk.get("f0"),
                "secondary": pk.get("secondary", []),
                "bedrock_depth": bd.get("depth", 0),
                "vs30": bd.get("vs30"),
                "vsavg": bd.get("vsavg"),
            }
        return result

    def set_auto_peak_config(self, cfg):
        """Set auto-peak detection configuration from dialog."""
        self._auto_peak_cfg = cfg

    # ══════════════════════════════════════════════════════════════
    #  NAVIGATION
    # ══════════════════════════════════════════════════════════════
    def _on_list_row(self, row):
        if 0 <= row < len(self._steps):
            self._current_idx = row
            self._show_step(row)
            self._update_nav()

    def _go_prev(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._list.setCurrentRow(self._current_idx)

    def _go_next(self):
        if self._current_idx < len(self._steps) - 1:
            self._current_idx += 1
            self._list.setCurrentRow(self._current_idx)

    def _update_nav(self):
        n = len(self._steps)
        self._btn_prev.setEnabled(self._current_idx > 0)
        self._btn_next.setEnabled(self._current_idx < n - 1)
        self._info.setText(f"{self._current_idx + 1} / {n}")

    def _on_finish(self):
        """Emit wizard_finished with all peak/bedrock data."""
        self.wizard_finished.emit(self.get_peak_data())

    # ══════════════════════════════════════════════════════════════
    #  DISPLAY
    # ══════════════════════════════════════════════════════════════
    def _show_step(self, idx):
        if idx < 0 or idx >= len(self._steps):
            return
        step = self._steps[idx]
        name = step["name"]
        n_layers = self._parse_n_layers(name)
        self._title_label.setText(f"Step: {name}")
        self._step_info.setText(
            f"{n_layers} layers" if n_layers else "")
        self._redraw_hv()
        self._populate_bedrock_combo()
        self._redraw_vs()

    def _redraw_hv(self):
        """Redraw HV curve for current step with peak markers."""
        if not self._steps:
            return
        step = self._steps[self._current_idx]
        freqs, amps = step.get("freqs"), step.get("amps")
        name = step["name"]

        fig = self._hv_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if freqs is None or amps is None:
            ax.text(0.5, 0.5, "No HV data for this step",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="gray")
            fig.tight_layout()
            self._hv_plot.refresh()
            return

        ax.plot(freqs, amps, color="royalblue", lw=1.8, label="H/V")
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("H/V Ratio")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

        # Draw peaks
        pk = self._peak_data.get(name, {})
        f0 = pk.get("f0")
        if f0:
            ax.plot(f0[0], f0[1], "*", color="red", ms=14, zorder=10,
                    markeredgecolor="darkred", markeredgewidth=0.8)
            ax.axvline(f0[0], color="red", ls="--", lw=0.8, alpha=0.4)
            ann_f0 = ax.annotate(
                f"f0 = {f0[0]:.4f} Hz\nA = {f0[1]:.2f}",
                xy=(f0[0], f0[1]), xytext=(12, -18),
                textcoords="offset points", fontsize=8, color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="red", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
            ann_f0.draggable(True)

        sec = pk.get("secondary", [])
        for j, s in enumerate(sec):
            sc = SEC_COLORS[j % len(SEC_COLORS)]
            ax.plot(s[0], s[1], "*", color=sc, ms=11, zorder=9,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.axvline(s[0], color=sc, ls=":", lw=0.7, alpha=0.4)
            y_off = 10 if j % 2 == 0 else -18
            ann_sec = ax.annotate(
                f"Sec.{j+1}: {s[0]:.3f} Hz ({s[1]:.2f})",
                xy=(s[0], s[1]), xytext=(8, y_off),
                textcoords="offset points", fontsize=7, color=sc,
                bbox=dict(boxstyle="round,pad=0.2", fc="#f0f0f0",
                          ec=sc, alpha=0.8))
            ann_sec.draggable(True)

        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        fig.tight_layout()
        self._hv_plot.refresh()
        self._update_sel_label()

    def _redraw_vs(self):
        """Redraw Vs mini profile for current step."""
        if not self._steps or not self._chk_show_vs.isChecked():
            return
        step = self._steps[self._current_idx]
        layers = step.get("layers")
        name = step["name"]

        fig = self._vs_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if not layers:
            ax.text(0.5, 0.5, "No model", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            fig.tight_layout()
            self._vs_plot.refresh()
            return

        finite = [L for L in layers if not L.get("is_halfspace", False)]
        hs = [L for L in layers if L.get("is_halfspace", False)]

        depths, vs_vals = [], []
        z = 0.0
        for L in finite:
            depths.append(z); vs_vals.append(L["vs"])
            z += L["thickness"]
            depths.append(z); vs_vals.append(L["vs"])
        total = z

        if hs:
            hd = total * 0.25
            depths.append(z); vs_vals.append(hs[0]["vs"])
            z += hd
            depths.append(z); vs_vals.append(hs[0]["vs"])

        ax.plot(vs_vals, depths, color="teal", lw=1.5)
        ax.fill_betweenx(depths, 0, vs_vals, alpha=0.1, color="teal")
        ax.axhline(total, color="red", lw=0.8, ls="--", alpha=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=7)
        ax.set_title(f"{len(finite)}L", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.yaxis.tick_right()
        ax.grid(True, alpha=0.2)

        # Bedrock line
        bd = self._bedrock_data.get(name, {})
        bedrock_depth = bd.get("depth", total)
        if bedrock_depth and bedrock_depth < z:
            ax.axhline(bedrock_depth, color="brown", lw=1.2, ls="-.",
                        alpha=0.7, label=f"Bedrock @ {bedrock_depth:.1f}m")
            ax.legend(fontsize=5, loc="lower right")

        # Vs30 annotation
        vs30 = bd.get("vs30")
        if vs30 and self._chk_vs30.isChecked():
            ax.axhline(30.0, color="blue", lw=0.8, ls="-.", alpha=0.5)
            vs_max = max(vs_vals) if vs_vals else 500
            ax.annotate(f"Vs30={vs30:.0f}", xy=(vs_max * 0.5, 30.0),
                        xytext=(0, -8), textcoords="offset points",
                        fontsize=6, color="blue", fontweight="bold")

        # VsAvg annotation
        vsavg = bd.get("vsavg")
        if vsavg and self._chk_vsavg.isChecked():
            ax.annotate(f"VsAvg={vsavg:.0f} m/s",
                        xy=(0.02, 0.02), xycoords="axes fraction",
                        fontsize=6, color="teal", fontweight="bold")

        # Vs info label
        parts = []
        if vs30:
            parts.append(f"Vs30 = {vs30:.0f} m/s")
        if vsavg:
            parts.append(f"VsAvg = {vsavg:.0f} m/s")
        parts.append(f"Bedrock @ {bedrock_depth:.1f} m")
        self._vs_label.setText(" | ".join(parts))

        fig.tight_layout()
        self._vs_plot.refresh()

    # ══════════════════════════════════════════════════════════════
    #  PEAK PICKING
    # ══════════════════════════════════════════════════════════════
    def _toggle_f0(self, on):
        self._pick_f0 = on
        if on:
            self._btn_sec.setChecked(False)

    def _toggle_sec(self, on):
        self._pick_sec = on
        if on:
            self._btn_f0.setChecked(False)

    def _clear_peaks(self):
        if not self._steps:
            return
        name = self._steps[self._current_idx]["name"]
        self._peak_data[name] = {"f0": None, "secondary": []}
        self._update_list_icon(self._current_idx, False)
        self._redraw_hv()

    def _on_press(self, event):
        """Record press position for drag-to-select."""
        if event.inaxes is None or not self._steps:
            return
        if not (self._pick_f0 or self._pick_sec):
            return
        if self._hv_plot.toolbar.mode:
            return
        self._drag_start_x = event.xdata

    def _on_motion(self, event):
        """Show drag rectangle during click-and-release selection."""
        if self._drag_start_x is None or event.inaxes is None:
            return
        if not (self._pick_f0 or self._pick_sec):
            return

        ax = event.inaxes
        # Remove previous rectangle
        if self._drag_rect is not None:
            try:
                self._drag_rect.remove()
            except ValueError:
                pass
            self._drag_rect = None

        x0 = min(self._drag_start_x, event.xdata)
        x1 = max(self._drag_start_x, event.xdata)
        color = "red" if self._pick_f0 else "orange"
        self._drag_rect = ax.axvspan(x0, x1, alpha=0.15, color=color)
        self._hv_plot.canvas.draw_idle()

    def _on_release(self, event):
        """Handle release: drag (range) or click (single point)."""
        if event.inaxes is None or not self._steps:
            self._drag_start_x = None
            return
        if not (self._pick_f0 or self._pick_sec):
            self._drag_start_x = None
            return
        if self._hv_plot.toolbar.mode:
            self._drag_start_x = None
            return

        step = self._steps[self._current_idx]
        freqs, amps = step.get("freqs"), step.get("amps")
        if freqs is None:
            self._drag_start_x = None
            return

        name = step["name"]
        DRAG_THRESHOLD = 0.02  # relative to frequency range

        if self._drag_start_x is not None:
            drag_dist = abs(event.xdata - self._drag_start_x)
            freq_range = freqs[-1] - freqs[0]
            is_drag = drag_dist > freq_range * DRAG_THRESHOLD

            if is_drag:
                # Drag: find peak within the selected range
                x0 = min(self._drag_start_x, event.xdata)
                x1 = max(self._drag_start_x, event.xdata)
                mask = (freqs >= x0) & (freqs <= x1)
                if np.any(mask):
                    masked = np.where(mask, amps, -np.inf)
                    idx = int(np.argmax(masked))
                    f, a = float(freqs[idx]), float(amps[idx])
                    if self._pick_f0:
                        self._peak_data[name]["f0"] = (f, a, idx)
                        self._btn_f0.setChecked(False)
                        self._update_list_icon(self._current_idx, True)
                    elif self._pick_sec:
                        self._peak_data[name].setdefault(
                            "secondary", []).append((f, a, idx))
                    self._redraw_hv()
            else:
                # Single click: interpolate at clicked position
                cx = event.xdata
                amp_interp = float(np.interp(cx, freqs, amps))
                idx = int(np.argmin(np.abs(freqs - cx)))
                f, a = float(cx), amp_interp

                if self._pick_f0:
                    self._peak_data[name]["f0"] = (f, a, idx)
                    self._btn_f0.setChecked(False)
                    self._update_list_icon(self._current_idx, True)
                elif self._pick_sec:
                    self._peak_data[name].setdefault(
                        "secondary", []).append((f, a, idx))
                self._redraw_hv()
        else:
            # Fallback single-click (no press recorded)
            cx = event.xdata
            amp_interp = float(np.interp(cx, freqs, amps))
            idx = int(np.argmin(np.abs(freqs - cx)))
            f, a = float(cx), amp_interp

            if self._pick_f0:
                self._peak_data[name]["f0"] = (f, a, idx)
                self._btn_f0.setChecked(False)
                self._update_list_icon(self._current_idx, True)
            elif self._pick_sec:
                self._peak_data[name].setdefault(
                    "secondary", []).append((f, a, idx))
            self._redraw_hv()

        # Clean up drag state
        self._drag_start_x = None
        if self._drag_rect is not None:
            try:
                self._drag_rect.remove()
            except ValueError:
                pass
            self._drag_rect = None

    def _update_sel_label(self):
        if not self._steps:
            return
        name = self._steps[self._current_idx]["name"]
        pk = self._peak_data.get(name, {})
        parts = []
        f0 = pk.get("f0")
        if f0:
            parts.append(f"f0 = {f0[0]:.4f} Hz (A={f0[1]:.2f})")
        for j, s in enumerate(pk.get("secondary", [])):
            parts.append(f"Sec.{j+1} = {s[0]:.3f} Hz")
        bd = self._bedrock_data.get(name, {})
        if bd.get("vs30"):
            parts.append(f"Vs30={bd['vs30']:.0f}")
        self._sel_label.setText("  |  ".join(parts) if parts else
                                "Click on the curve to select peaks")

    def _update_list_icon(self, idx, has_peak):
        if 0 <= idx < self._list.count():
            name = self._step_names[idx]
            icon = "✓" if has_peak else "  "
            self._list.item(idx).setText(f"{icon}  {name}")

    # ══════════════════════════════════════════════════════════════
    #  BEDROCK INTERFACE
    # ══════════════════════════════════════════════════════════════
    def _populate_bedrock_combo(self):
        """Populate bedrock dropdown with layer interfaces for current step."""
        if not self._steps:
            return
        step = self._steps[self._current_idx]
        layers = step.get("layers")
        name = step["name"]

        self._bedrock_combo.blockSignals(True)
        self._bedrock_combo.clear()

        if not layers:
            self._bedrock_combo.blockSignals(False)
            return

        finite = [L for L in layers if not L.get("is_halfspace", False)]
        z = 0.0
        for i, L in enumerate(finite):
            z += L["thickness"]
            self._bedrock_combo.addItem(
                f"Interface {i+1}: {z:.1f} m (Vs={L['vs']:.0f})",
                z)

        # Select current bedrock depth
        bd = self._bedrock_data.get(name, {})
        current_depth = bd.get("depth", z)
        best_idx = self._bedrock_combo.count() - 1
        for i in range(self._bedrock_combo.count()):
            if abs(self._bedrock_combo.itemData(i) - current_depth) < 0.01:
                best_idx = i
                break
        self._bedrock_combo.setCurrentIndex(best_idx)
        self._bedrock_combo.blockSignals(False)

    def _on_bedrock_changed(self):
        """Recompute Vs30/VsAvg when user changes bedrock interface."""
        if not self._steps:
            return
        step = self._steps[self._current_idx]
        name = step["name"]
        layers = step.get("layers")
        if not layers:
            return

        idx = self._bedrock_combo.currentIndex()
        if idx < 0:
            return
        bedrock_depth = self._bedrock_combo.itemData(idx)
        if bedrock_depth is None:
            return

        # Compute Vs30 and VsAvg to bedrock depth
        vs30, vsavg = None, None
        try:
            layer_tuples = [(L["thickness"], L["vs"]) for L in layers
                           if not L.get("is_halfspace", False)]
            hs = [L for L in layers if L.get("is_halfspace", False)]
            if hs:
                layer_tuples.append((0, hs[0]["vs"]))

            from ..core.vs_average import compute_vs_average
            # Vs30
            res30 = compute_vs_average(layer_tuples, target_depth=30.0,
                                       use_halfspace=True)
            vs30 = res30.vs_avg

            # VsAvg to bedrock
            if bedrock_depth > 0:
                res_br = compute_vs_average(layer_tuples,
                                           target_depth=bedrock_depth,
                                           use_halfspace=False)
                vsavg = res_br.vs_avg
        except Exception:
            pass

        self._bedrock_data[name] = {
            "depth": bedrock_depth,
            "vs30": vs30,
            "vsavg": vsavg,
        }
        self._redraw_vs()
        self._update_sel_label()

    # ══════════════════════════════════════════════════════════════
    #  AUTO PEAK DETECTION
    # ══════════════════════════════════════════════════════════════
    def _run_auto_detection(self):
        """Run auto peak detection on all steps using cfg."""
        cfg = self._auto_peak_cfg or {}
        min_prom = cfg.get("min_prominence", 0.3)
        min_amp = cfg.get("min_amplitude", 1.5)
        n_secondary = cfg.get("n_secondary", 1)
        ranges = cfg.get("ranges", [])

        for i, step in enumerate(self._steps):
            freqs, amps = step.get("freqs"), step.get("amps")
            name = step["name"]
            if freqs is None or amps is None:
                continue

            # Primary peak detection
            f0 = self._detect_primary(freqs, amps, ranges)
            sec = self._detect_secondary(freqs, amps, f0, n_secondary,
                                         min_prom, min_amp, ranges)
            self._peak_data[name] = {"f0": f0, "secondary": sec}
            self._update_list_icon(i, f0 is not None)

        self._redraw_hv()

    def _detect_primary(self, freqs, amps, ranges):
        """Detect primary peak, honouring ranges[0] if configured."""
        rng = ranges[0] if ranges else None
        if rng:
            fmin_r = rng.get("min", 0.0)
            fmax_r = rng.get("max", 999.0)
            mask = (freqs >= fmin_r) & (freqs <= fmax_r)
            if np.any(mask):
                masked = np.where(mask, amps, -np.inf)
                idx = int(np.argmax(masked))
                return (float(freqs[idx]), float(amps[idx]), idx)
        # Fallback: global max
        idx = int(np.argmax(amps))
        return (float(freqs[idx]), float(amps[idx]), idx)

    def _detect_secondary(self, freqs, amps, f0, n_secondary,
                          min_prom, min_amp, ranges):
        """Detect secondary peaks using ranges or scipy find_peaks."""
        sec = []
        if f0 is None:
            return sec

        for si in range(n_secondary):
            rng = ranges[si + 1] if (si + 1) < len(ranges) else None
            if rng:
                fmin_r = rng.get("min", 0.0)
                fmax_r = rng.get("max", 999.0)
                mask = (freqs >= fmin_r) & (freqs <= fmax_r)
                # Exclude primary peak region
                f0_f = f0[0]
                exclusion = max(f0_f * 0.1, 0.05)
                mask &= ~((freqs >= f0_f - exclusion) &
                          (freqs <= f0_f + exclusion))
                # Exclude already-found secondary peaks
                for prev in sec:
                    exc = max(prev[0] * 0.1, 0.05)
                    mask &= ~((freqs >= prev[0] - exc) &
                              (freqs <= prev[0] + exc))
                if np.any(mask):
                    masked = np.where(mask, amps, -np.inf)
                    idx = int(np.argmax(masked))
                    if amps[idx] >= min_amp:
                        sec.append((float(freqs[idx]), float(amps[idx]), idx))
            else:
                # Scipy-based detection
                try:
                    from scipy.signal import find_peaks as _find_peaks
                    peak_indices, props = _find_peaks(amps, prominence=min_prom)
                    # Filter by min amplitude and exclude primary
                    candidates = []
                    for pi in peak_indices:
                        if amps[pi] < min_amp:
                            continue
                        if abs(freqs[pi] - f0[0]) < max(f0[0] * 0.1, 0.05):
                            continue
                        skip = False
                        for prev in sec:
                            if abs(freqs[pi] - prev[0]) < max(prev[0] * 0.1, 0.05):
                                skip = True
                                break
                        if not skip:
                            candidates.append((float(freqs[pi]),
                                               float(amps[pi]), int(pi)))
                    # Sort by amplitude descending, take next
                    candidates.sort(key=lambda x: -x[1])
                    if candidates:
                        sec.append(candidates[0])
                except ImportError:
                    pass

        return sec

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════
    def _toggle_vs_panel(self, show):
        self._vs_panel.setVisible(show)
        if show:
            self._redraw_vs()

    def _load_model_layers(self, model_file):
        """Load layer data from an HVf model file into a list of dicts."""
        if not model_file:
            return []
        try:
            from pathlib import Path
            p = Path(str(model_file))
            if not p.exists():
                return []
            lines = p.read_text().strip().split("\n")
            if not lines:
                return []
            n = int(lines[0].strip())
            layers = []
            for line in lines[1:n + 1]:
                parts = line.strip().split()
                if len(parts) >= 4:
                    h, vp, vs, rho = (float(parts[0]), float(parts[1]),
                                      float(parts[2]), float(parts[3]))
                    layers.append({
                        "thickness": h,
                        "vp": vp,
                        "vs": vs,
                        "density": rho,
                        "is_halfspace": h == 0,
                    })
            return layers
        except Exception:
            return []

    @staticmethod
    def _parse_n_layers(step_name):
        """Extract layer count from step name like 'Step0_5-layer'."""
        try:
            for part in step_name.split("_"):
                if "-layer" in part:
                    return int(part.split("-")[0])
        except (ValueError, IndexError):
            pass
        return None
