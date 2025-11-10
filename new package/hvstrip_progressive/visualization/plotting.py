"""
Advanced plotting utilities for HVSR analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns


class HVSRPlotter:
    """Advanced plotting class for HVSR visualization."""
    
    def __init__(self, style='publication', colormap='viridis'):
        """
        Initialize plotter.
        
        Parameters:
            style: Plot style ('publication', 'presentation', 'minimal')
            colormap: Matplotlib colormap name
        """
        self.style = style
        self.colormap = colormap
        self._setup_style()
    
    def _setup_style(self):
        """Configure plotting style."""
        if self.style == 'publication':
            sns.set_style("whitegrid")
            self.figsize = (8, 6)
        elif self.style == 'presentation':
            sns.set_style("darkgrid") 
            self.figsize = (12, 8)
        else:  # minimal
            sns.set_style("white")
            self.figsize = (6, 4)
    
    def plot_multiple_curves(self, curves_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                           title: str = "HVSR Comparison",
                           save_path: Optional[Path] = None) -> plt.Figure:
        """Plot multiple HVSR curves for comparison."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_curves = len(curves_data)
        colors = plt.cm.get_cmap(self.colormap)(np.linspace(0, 1, n_curves))
        
        for i, (label, (freqs, amps)) in enumerate(curves_data.items()):
            ax.semilogx(freqs, amps, color=colors[i], linewidth=2, 
                       label=label, alpha=0.8)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('H/V Amplitude', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_comparison_plot(data_dict: Dict[str, Dict], 
                         output_path: Optional[Path] = None) -> plt.Figure:
    """Create comparison plot from multiple analysis results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    labels = list(data_dict.keys())
    n_datasets = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    
    # Panel 1: HV curves overlay
    for i, (label, data) in enumerate(data_dict.items()):
        if 'frequencies' in data and 'amplitudes' in data:
            ax1.semilogx(data['frequencies'], data['amplitudes'], 
                        color=colors[i], linewidth=2, label=label, alpha=0.8)
    
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('H/V Amplitude')
    ax1.set_title('(a) HVSR Curves Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Peak frequencies
    peak_freqs = [data.get('peak_frequency', 0) for data in data_dict.values()]
    ax2.bar(range(len(labels)), peak_freqs, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel('Peak Frequency (Hz)')
    ax2.set_title('(b) Peak Frequencies')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Peak amplitudes
    peak_amps = [data.get('peak_amplitude', 0) for data in data_dict.values()]
    ax3.bar(range(len(labels)), peak_amps, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45)
    ax3.set_ylabel('Peak Amplitude')
    ax3.set_title('(c) Peak Amplitudes')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary statistics
    ax4.axis('off')
    summary_text = "Comparison Summary\n" + "="*20 + "\n"
    summary_text += f"Datasets: {n_datasets}\n"
    summary_text += f"Freq range: {min(peak_freqs):.2f}-{max(peak_freqs):.2f} Hz\n"
    summary_text += f"Amp range: {min(peak_amps):.1f}-{max(peak_amps):.1f}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('HVSR Analysis Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


__all__ = [
    "HVSRPlotter",
    "create_comparison_plot"
]
