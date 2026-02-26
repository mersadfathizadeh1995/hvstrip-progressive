"""
Profile preview widget showing Vs profile visualization.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from ...core.soil_profile import SoilProfile, compute_halfspace_display_depth


class ProfilePreviewWidget(QWidget):
    """
    Widget for displaying a Vs depth profile visualization.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._profile = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(4, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        
        layout.addWidget(self.canvas)
        self.setMinimumWidth(300)
        self.setMinimumHeight(400)
        
        self._draw_empty()
    
    def _draw_empty(self):
        """Draw empty plot with instructions."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5, 0.5, "Add layers to see\nVs profile preview",
            ha='center', va='center', fontsize=12, color='gray',
            transform=ax.transAxes
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        self.canvas.draw()
    
    def set_profile(self, profile: SoilProfile):
        """Set the profile to display."""
        self._profile = profile
        self._update_plot()
    
    def _update_plot(self):
        """Update the Vs profile plot."""
        if self._profile is None or len(self._profile.layers) == 0:
            self._draw_empty()
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        depths = [0]
        vs_values = []
        
        total_finite = sum(
            l.thickness for l in self._profile.layers if not l.is_halfspace
        )
        for layer in self._profile.layers:
            if layer.is_halfspace:
                depths.append(
                    depths[-1] + compute_halfspace_display_depth(total_finite)
                )
            else:
                depths.append(depths[-1] + layer.thickness)
            vs_values.append(layer.vs)
        
        plot_depths = []
        plot_vs = []
        
        for i, vs in enumerate(vs_values):
            plot_depths.extend([depths[i], depths[i + 1]])
            plot_vs.extend([vs, vs])
        
        ax.plot(plot_vs, plot_depths, 'b-', linewidth=2, label='Vs')
        
        for i, layer in enumerate(self._profile.layers):
            mid_depth = (depths[i] + depths[i + 1]) / 2
            ax.axhline(
                y=depths[i + 1], 
                color='gray', 
                linestyle='--', 
                linewidth=0.5, 
                alpha=0.5
            )
            
            if layer.is_halfspace:
                label = f"HS: Vs={layer.vs:.0f}"
            else:
                label = f"L{i+1}: Vs={layer.vs:.0f}"
            
            ax.annotate(
                label,
                xy=(layer.vs, mid_depth),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8,
                va='center'
            )
        
        ax.set_xlabel('Vs (m/s)', fontsize=10)
        ax.set_ylabel('Depth (m)', fontsize=10)
        ax.set_title('Vs Profile Preview', fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=depths[-1] * 1.05, top=0)
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def refresh(self):
        """Refresh the plot."""
        self._update_plot()
