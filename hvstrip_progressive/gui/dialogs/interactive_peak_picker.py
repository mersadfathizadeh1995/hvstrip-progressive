"""
Interactive Peak Picker Dialog for HVSR Progressive Layer Stripping.

Allows users to manually select peaks by clicking on the HVSR curve
for each stripping step.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QMessageBox, QSplitter, QListWidget, QListWidgetItem,
    QWidget
)
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class InteractivePeakPickerDialog(QDialog):
    """Dialog for interactive peak selection across all stripping steps."""
    
    peaks_selected = Signal(dict)  # Emits {step_name: (freq, amp, idx)}
    
    def __init__(self, step_data: List[Dict], parent=None):
        """
        Initialize the interactive peak picker.
        
        Parameters
        ----------
        step_data : List[Dict]
            List of dictionaries containing step information:
            - 'name': Step folder name (e.g., 'Step1_10-layers')
            - 'folder': Path to step folder
            - 'freqs': numpy array of frequencies
            - 'amps': numpy array of amplitudes
            - 'model_file': Path to model file (optional)
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Interactive Peak Selection - Progressive Layer Stripping")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        self.step_data = step_data
        self.current_step_idx = 0
        self.selected_peaks: Dict[str, Tuple[float, float, int]] = {}
        self.undo_stack: List[Tuple[str, Optional[Tuple]]] = []
        
        self._setup_ui()
        self._load_step(0)
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Step list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        list_label = QLabel("Steps:")
        list_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(list_label)
        
        self.step_list = QListWidget()
        self.step_list.setMaximumWidth(200)
        for i, step in enumerate(self.step_data):
            item = QListWidgetItem(step['name'])
            self.step_list.addItem(item)
        self.step_list.currentRowChanged.connect(self._on_step_selected)
        left_layout.addWidget(self.step_list)
        
        splitter.addWidget(left_widget)
        
        # Right panel: Plot and controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 14px; padding: 5px;")
        right_layout.addWidget(self.info_label)
        
        # Matplotlib figure - wider aspect ratio (14:5)
        self.figure = Figure(figsize=(14, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 1)
        
        # Connect click event
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        
        # Pick mode controls
        pick_layout = QHBoxLayout()
        
        self.btn_pick_mode = QPushButton("Click Selection: ON")
        self.btn_pick_mode.setCheckable(True)
        self.btn_pick_mode.setChecked(True)  # Start in pick mode
        self.btn_pick_mode.setStyleSheet(
            "QPushButton:checked { background-color: #107c10; color: white; font-weight: bold; padding: 10px 20px; }"
            "QPushButton { padding: 10px 20px; background-color: #d83b01; color: white; }"
        )
        self.btn_pick_mode.clicked.connect(self._toggle_pick_mode)
        pick_layout.addWidget(self.btn_pick_mode)
        
        # Toggle Vs Profile button
        self.btn_show_vs = QPushButton("Show Vs Profile")
        self.btn_show_vs.setCheckable(True)
        self.btn_show_vs.setChecked(True)  # Start with Vs visible
        self.show_vs_profile = True
        self.btn_show_vs.setStyleSheet(
            "QPushButton:checked { background-color: #0078d4; color: white; padding: 10px 15px; }"
            "QPushButton { padding: 10px 15px; background-color: #666; color: white; }"
        )
        self.btn_show_vs.clicked.connect(self._toggle_vs_profile)
        pick_layout.addWidget(self.btn_show_vs)
        
        pick_layout.addStretch()
        right_layout.addLayout(pick_layout)
        
        # Selection info - must be created BEFORE _set_pick_mode is called
        self.selection_label = QLabel("Click on the curve to select a peak")
        self.selection_label.setStyleSheet(
            "font-size: 12px; color: #0078d4; padding: 5px; "
            "background-color: #f0f0f0; border-radius: 3px;"
        )
        right_layout.addWidget(self.selection_label)
        
        # Now enable pick mode (after selection_label exists)
        self._set_pick_mode(True)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.clicked.connect(self._go_previous)
        btn_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self._go_next)
        btn_layout.addWidget(self.btn_next)
        
        btn_layout.addStretch()
        
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self._undo)
        self.btn_undo.setEnabled(False)
        btn_layout.addWidget(self.btn_undo)
        
        self.btn_auto = QPushButton("Auto-detect")
        self.btn_auto.setToolTip("Automatically detect peak for this step")
        self.btn_auto.clicked.connect(self._auto_detect_current)
        btn_layout.addWidget(self.btn_auto)
        
        self.btn_skip = QPushButton("Skip")
        self.btn_skip.setToolTip("Skip this step (use auto-detected peak)")
        self.btn_skip.clicked.connect(self._skip_current)
        btn_layout.addWidget(self.btn_skip)
        
        btn_layout.addStretch()
        
        self.btn_finish = QPushButton("Finish")
        self.btn_finish.setStyleSheet(
            "background-color: #107c10; color: white; padding: 8px 20px;"
        )
        self.btn_finish.clicked.connect(self._finish)
        btn_layout.addWidget(self.btn_finish)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        
        right_layout.addLayout(btn_layout)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([150, 850])
        
        layout.addWidget(splitter)
        
        self._update_buttons()
    
    def _toggle_pick_mode(self):
        """Toggle peak selection mode."""
        self._set_pick_mode(self.btn_pick_mode.isChecked())
    
    def _toggle_vs_profile(self):
        """Toggle Vs profile visibility."""
        self.show_vs_profile = self.btn_show_vs.isChecked()
        if self.show_vs_profile:
            self.btn_show_vs.setText("Show Vs Profile")
        else:
            self.btn_show_vs.setText("Vs Profile: OFF")
        # Replot current step
        if self.current_step_idx < len(self.step_data):
            self._plot_step(self.step_data[self.current_step_idx])
    
    def _set_pick_mode(self, enabled: bool):
        """Enable or disable pick mode."""
        if enabled:
            # Disable any active toolbar mode
            if hasattr(self.toolbar, 'mode') and self.toolbar.mode:
                # Reset toolbar to normal mode
                if self.toolbar.mode == 'pan/zoom':
                    self.toolbar.pan()
                elif self.toolbar.mode == 'zoom rect':
                    self.toolbar.zoom()
            self.btn_pick_mode.setText("Click Selection: ON")
            self.selection_label.setText("Click anywhere on the blue curve to select that point as f0")
        else:
            self.btn_pick_mode.setText("Click Selection: OFF")
            self.selection_label.setText("Selection disabled - use toolbar to pan/zoom, then turn selection ON")
    
    def _load_step(self, idx: int):
        """Load and display a specific step."""
        if idx < 0 or idx >= len(self.step_data):
            return
        
        self.current_step_idx = idx
        step = self.step_data[idx]
        
        # Update info label
        n_selected = len(self.selected_peaks)
        self.info_label.setText(
            f"Step {idx + 1} of {len(self.step_data)}: {step['name']}  |  "
            f"Peaks selected: {n_selected}/{len(self.step_data)}"
        )
        
        # Update list selection
        self.step_list.setCurrentRow(idx)
        
        # Plot the curve
        self._plot_step(step)
        
        # Update buttons
        self._update_buttons()
    
    def _plot_step(self, step: Dict):
        """Plot the HVSR curve and optionally Vs profile for a step."""
        self.figure.clear()
        
        freqs = step.get('freqs', np.array([]))
        amps = step.get('amps', np.array([]))
        step_name = step['name']
        
        if len(freqs) == 0 or len(amps) == 0:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return
        
        # Check if Vs profile should be shown
        if self.show_vs_profile:
            # Use gridspec for unequal widths: HV gets 80%, Vs gets 20%
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.25)
            ax_hv = self.figure.add_subplot(gs[0])
            ax_vs = self.figure.add_subplot(gs[1])
        else:
            # Full width for HV curve only
            ax_hv = self.figure.add_subplot(111)
            ax_vs = None
        
        # === HVSR Curve (main panel) ===
        ax_hv.semilogx(freqs, amps, 'b-', linewidth=2, label='H/V Ratio')
        
        # Mark auto-detected peak (gray)
        auto_peak_idx = np.argmax(amps)
        auto_peak_f = freqs[auto_peak_idx]
        auto_peak_a = amps[auto_peak_idx]
        ax_hv.scatter(auto_peak_f, auto_peak_a, color='gray', s=100, 
                     marker='o', alpha=0.5, label=f'Auto: {auto_peak_f:.2f} Hz', zorder=4)
        
        # Mark selected peak (red) if exists
        sel_f = None
        if step_name in self.selected_peaks:
            sel_f, sel_a, sel_idx = self.selected_peaks[step_name]
            ax_hv.scatter(sel_f, sel_a, color='red', s=200, marker='*', 
                         edgecolors='darkred', linewidth=1.5,
                         label=f'Selected: {sel_f:.2f} Hz', zorder=5)
            ax_hv.axvline(x=sel_f, color='red', linestyle='--', alpha=0.5)
        
        ax_hv.set_xlabel('Frequency (Hz)', fontsize=12)
        ax_hv.set_ylabel('H/V Amplitude Ratio', fontsize=12)
        ax_hv.set_title(f'HVSR Curve - {step_name}', fontsize=13, fontweight='bold')
        ax_hv.grid(True, alpha=0.3, which='both')
        ax_hv.legend(loc='upper right', fontsize=10)
        ax_hv.set_xlim(freqs[0], freqs[-1])
        
        # === Vs Profile (small side panel) ===
        if ax_vs is not None:
            model_file = step.get('model_file')
            if model_file and Path(model_file).exists():
                try:
                    self._plot_vs_profile(ax_vs, model_file, step_name, sel_f)
                except Exception as e:
                    ax_vs.text(0.5, 0.5, f"Error:\n{e}", 
                              ha='center', va='center', transform=ax_vs.transAxes, fontsize=9)
                    ax_vs.set_title('Vs', fontsize=11, fontweight='bold')
            else:
                ax_vs.text(0.5, 0.5, "No model", 
                          ha='center', va='center', transform=ax_vs.transAxes, fontsize=10)
                ax_vs.set_title('Vs', fontsize=11, fontweight='bold')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Update selection label
        self._update_selection_label(step_name)
    
    def _plot_vs_profile(self, ax, model_file: Path, step_name: str, selected_freq: float = None):
        """Plot Vs profile with layer information and depth annotations (compact version)."""
        model_file = Path(model_file)
        
        # Load model data
        data = np.loadtxt(model_file, skiprows=1)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Extract layers: thickness, Vp, Vs, density
        thicknesses = data[:, 0]
        vs_values = data[:, 2]
        
        # Calculate depths
        depths = np.zeros(len(thicknesses) + 1)
        for i, h in enumerate(thicknesses):
            if h > 0:
                depths[i + 1] = depths[i] + h
            else:
                # Half-space: extend to reasonable depth
                depths[i + 1] = depths[i] + max(100, depths[i] * 0.3)
        
        n_layers = len(vs_values)
        
        # Create step profile for plotting
        plot_depths = []
        plot_vs = []
        
        for i in range(len(vs_values)):
            plot_depths.extend([depths[i], depths[i + 1]])
            plot_vs.extend([vs_values[i], vs_values[i]])
        
        # Plot the Vs profile
        ax.fill_betweenx(plot_depths, 0, plot_vs, alpha=0.3, color='teal')
        ax.step(plot_vs + [plot_vs[-1]], [0] + plot_depths, where='pre', 
               color='teal', linewidth=1.5, linestyle='-')
        
        # Highlight the deepest interface being stripped
        if n_layers > 0 and len(depths) > 2:
            deepest_depth = depths[-2]
            ax.axhline(y=deepest_depth, color='red', linestyle='-', alpha=0.7, linewidth=1.5)
            # Compact depth annotation
            ax.text(0.95, 0.02, f'{deepest_depth:.0f}m', 
                   transform=ax.transAxes, fontsize=8, fontweight='bold',
                   color='red', ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Vs', fontsize=9)
        ax.set_ylabel('Depth (m)', fontsize=9)
        ax.set_title(f'{n_layers}L', fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        if plot_vs:
            ax.set_xlim(0, max(plot_vs) * 1.1)
    
    def _update_selection_label(self, step_name: str):
        """Update the selection label based on current selection."""
        if step_name in self.selected_peaks:
            sel_f, sel_a, _ = self.selected_peaks[step_name]
            self.selection_label.setText(
                f"Selected: f0 = {sel_f:.3f} Hz, Amplitude = {sel_a:.2f}"
            )
            self.selection_label.setStyleSheet(
                "font-size: 13px; color: #107c10; padding: 8px; font-weight: bold; "
                "background-color: #e6ffe6; border-radius: 5px;"
            )
        else:
            self.selection_label.setText("Click anywhere on the blue HVSR curve to select f0")
            self.selection_label.setStyleSheet(
                "font-size: 13px; color: #0078d4; padding: 8px; "
                "background-color: #f0f0f0; border-radius: 5px;"
            )
    
    def _on_canvas_click(self, event):
        """Handle click on the matplotlib canvas."""
        # Check if pick mode is enabled
        if not self.btn_pick_mode.isChecked():
            return
        
        # Check if click is inside axes
        if event.inaxes is None:
            return
        
        # Only handle left click
        if event.button != 1:
            return
        
        # Check if toolbar navigation mode is active (pan/zoom)
        # If so, don't process the click as peak selection
        try:
            toolbar_mode = getattr(self.toolbar, 'mode', '')
            if toolbar_mode and toolbar_mode != '':
                return
        except Exception:
            pass
        
        # Validate click data
        if event.xdata is None or event.ydata is None:
            return
        
        step = self.step_data[self.current_step_idx]
        freqs = step.get('freqs', np.array([]))
        amps = step.get('amps', np.array([]))
        
        if len(freqs) == 0:
            return
        
        click_freq = event.xdata
        
        # Ensure click is within frequency range
        if click_freq < freqs.min() or click_freq > freqs.max():
            return
        
        try:
            # Find nearest point on the curve to clicked frequency
            peak_idx = self._find_nearest_point(freqs, amps, click_freq)
            peak_freq = freqs[peak_idx]
            peak_amp = amps[peak_idx]
            
            # Store previous selection for undo
            step_name = step['name']
            prev_selection = self.selected_peaks.get(step_name)
            self.undo_stack.append((step_name, prev_selection))
            
            # Update selection
            self.selected_peaks[step_name] = (float(peak_freq), float(peak_amp), int(peak_idx))
            
            # Update list item to show it's selected
            self._update_list_item(self.current_step_idx, selected=True)
            
            # Replot
            self._plot_step(step)
            
            # Enable undo
            self.btn_undo.setEnabled(True)
            
        except Exception as e:
            print(f"Error selecting peak: {e}")
    
    def _find_nearest_point(self, freqs: np.ndarray, amps: np.ndarray, 
                             target_freq: float) -> int:
        """Find the nearest data point to the target frequency.
        
        This allows selecting ANY point on the curve, not just local peaks.
        """
        # Use log scale for distance calculation since x-axis is logarithmic
        log_freqs = np.log10(freqs)
        log_target = np.log10(target_freq)
        
        # Find the index of the nearest frequency
        distances = np.abs(log_freqs - log_target)
        nearest_idx = int(np.argmin(distances))
        
        return nearest_idx
    
    def _on_step_selected(self, row: int):
        """Handle step selection from list."""
        if row >= 0:
            self._load_step(row)
    
    def _go_previous(self):
        """Go to previous step."""
        if self.current_step_idx > 0:
            self._load_step(self.current_step_idx - 1)
    
    def _go_next(self):
        """Go to next step."""
        if self.current_step_idx < len(self.step_data) - 1:
            self._load_step(self.current_step_idx + 1)
    
    def _undo(self):
        """Undo the last peak selection."""
        if not self.undo_stack:
            return
        
        step_name, prev_selection = self.undo_stack.pop()
        
        if prev_selection is None:
            # Remove selection
            if step_name in self.selected_peaks:
                del self.selected_peaks[step_name]
        else:
            # Restore previous selection
            self.selected_peaks[step_name] = prev_selection
        
        # Find step index and update
        for i, step in enumerate(self.step_data):
            if step['name'] == step_name:
                self._update_list_item(i, selected=(step_name in self.selected_peaks))
                if i == self.current_step_idx:
                    self._plot_step(step)
                break
        
        self._update_buttons()
    
    def _auto_detect_current(self):
        """Auto-detect peak for current step."""
        step = self.step_data[self.current_step_idx]
        freqs = step.get('freqs', np.array([]))
        amps = step.get('amps', np.array([]))
        
        if len(freqs) == 0:
            return
        
        # Use global maximum as auto-detected peak
        peak_idx = int(np.argmax(amps))
        peak_freq = freqs[peak_idx]
        peak_amp = amps[peak_idx]
        
        # Store for undo
        step_name = step['name']
        prev_selection = self.selected_peaks.get(step_name)
        self.undo_stack.append((step_name, prev_selection))
        
        # Update selection
        self.selected_peaks[step_name] = (float(peak_freq), float(peak_amp), peak_idx)
        
        # Update UI
        self._update_list_item(self.current_step_idx, selected=True)
        self._plot_step(step)
        self.btn_undo.setEnabled(True)
    
    def _skip_current(self):
        """Skip current step and auto-detect, then move to next."""
        self._auto_detect_current()
        self._go_next()
    
    def _update_buttons(self):
        """Update button states."""
        self.btn_prev.setEnabled(self.current_step_idx > 0)
        self.btn_next.setEnabled(self.current_step_idx < len(self.step_data) - 1)
        self.btn_undo.setEnabled(len(self.undo_stack) > 0)
    
    def _update_list_item(self, idx: int, selected: bool):
        """Update list item appearance."""
        item = self.step_list.item(idx)
        if item:
            step_name = self.step_data[idx]['name']
            if selected:
                item.setText(f"✓ {step_name}")
            else:
                item.setText(step_name)
    
    def _finish(self):
        """Finish peak selection and emit results."""
        # Check for unselected steps
        unselected = []
        for step in self.step_data:
            if step['name'] not in self.selected_peaks:
                unselected.append(step['name'])
        
        if unselected:
            reply = QMessageBox.question(
                self,
                "Unselected Steps",
                f"{len(unselected)} steps have no peak selected.\n"
                "Use auto-detected peaks for these steps?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                # Auto-detect for unselected steps
                for step in self.step_data:
                    if step['name'] not in self.selected_peaks:
                        freqs = step.get('freqs', np.array([]))
                        amps = step.get('amps', np.array([]))
                        if len(freqs) > 0:
                            peak_idx = int(np.argmax(amps))
                            self.selected_peaks[step['name']] = (
                                float(freqs[peak_idx]),
                                float(amps[peak_idx]),
                                peak_idx
                            )
        
        # Emit results and close
        self.peaks_selected.emit(self.selected_peaks)
        self.accept()
    
    def get_selected_peaks(self) -> Dict[str, Tuple[float, float, int]]:
        """Return the selected peaks dictionary."""
        return self.selected_peaks
