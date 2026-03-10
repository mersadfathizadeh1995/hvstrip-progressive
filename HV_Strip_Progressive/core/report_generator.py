"""
Progressive Layer Stripping Analysis Report Generator
====================================================

Creates comprehensive scientific reports from batch_workflow.py results,
including publication-ready figures, data summaries, and analysis insights.

Author: HVSR-Diffuse Development Team
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import pandas as pd
from .soil_profile import compute_halfspace_display_depth


# Set publication-ready matplotlib defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 1.8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def _spread_annotations(annotations, min_gap_pts=14):
    """Nudge annotation offsets so labels don't overlap.

    *annotations* is a list of matplotlib ``Annotation`` objects that all
    use ``textcoords='offset points'``.  The function sorts them by their
    data-y coordinate and shifts each one whose offset point would be too
    close to a neighbour by alternating the text to above/below the data
    point.

    Parameters
    ----------
    annotations : list[matplotlib.text.Annotation]
    min_gap_pts : int
        Minimum vertical separation (in offset-points) between two labels.
    """
    if len(annotations) < 2:
        return
    # Sort by data-y (second element of xy) so we process bottom-to-top
    annotations.sort(key=lambda a: a.xy[1])
    for i in range(1, len(annotations)):
        prev = annotations[i - 1]
        curr = annotations[i]
        px, py = prev.xyann  # current offset
        cx, cy = curr.xyann
        # Rough overlap check: compare data-y + text offset
        prev_y = prev.xy[1] + py
        curr_y = curr.xy[1] + cy
        if abs(curr_y - prev_y) < min_gap_pts * 0.5:
            # Alternate: push current one above, previous below
            curr.xyann = (cx, abs(cy) + min_gap_pts)
            prev.xyann = (px, -(abs(py) + min_gap_pts))


class ProgressiveStrippingReporter:
    """Generate comprehensive reports from progressive layer stripping analysis."""
    
    def __init__(self, strip_directory: str, output_dir: Optional[str] = None):
        """
        Initialize the reporter.
        
        Parameters:
        -----------
        strip_directory : str
            Path to the 'strip' directory containing StepX_Y-layer folders
        output_dir : str, optional
            Output directory for reports (creates 'reports' if None)
        """
        self.strip_dir = Path(strip_directory)
        if not self.strip_dir.exists():
            raise FileNotFoundError(f"Strip directory not found: {self.strip_dir}")
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = self.strip_dir.parent / 'reports'
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect and analyze step data
        self.step_data = self._collect_step_data()
        self.analysis = self._analyze_data()
        
        print(f"[*] Initialized reporter with {len(self.step_data)} steps")
        print(f"[>] Output directory: {self.output_dir}")
    
    def _collect_step_data(self) -> List[Dict]:
        """Collect data from all step folders."""
        step_folders = []
        for folder in self.strip_dir.iterdir():
            if folder.is_dir() and folder.name.startswith('Step') and '_' in folder.name:
                step_folders.append(folder)
        
        # Sort by step number
        step_folders.sort(key=lambda x: int(x.name.split('_')[0].replace('Step', '')))
        
        step_data = []
        for folder in step_folders:
            data = self._parse_step_folder(folder)
            if data:
                step_data.append(data)
        
        return step_data
    
    def _parse_step_folder(self, folder: Path) -> Optional[Dict]:
        """Parse a single step folder to extract all data."""
        try:
            # Extract step info from folder name
            parts = folder.name.split('_')
            step_num = int(parts[0].replace('Step', ''))
            n_layers = int(parts[1].split('-')[0])
            
            data = {
                'folder': folder,
                'name': folder.name,
                'step': step_num,
                'n_finite_layers': n_layers
            }
            
            # Read velocity model
            model_files = list(folder.glob('model_*.txt'))
            if not model_files:
                return None
            
            data['model'] = self._read_velocity_model(model_files[0])
            
            # Read HV curve data
            hv_csv = folder / 'hv_curve.csv'
            if not hv_csv.exists():
                return None
            
            data['hv_data'] = self._read_hv_curve(hv_csv)
            
            # Read summary if available
            summary_files = list(folder.glob('*summary*.csv'))
            if summary_files:
                data['summary'] = self._read_summary(summary_files[0])
                # If summary provides peak info, prefer it for consistency with post-processing
                try:
                    summ = data.get('summary', {}) or {}
                    hv = data.get('hv_data', {}) or {}
                    pf = summ.get('Peak_Frequency_Hz')
                    pa = summ.get('Peak_Amplitude')
                    if pf is not None and pa is not None and hv:
                        freqs = hv.get('frequencies', [])
                        if len(freqs) > 0:
                            # Find nearest index to summary peak frequency
                            idx = int(np.argmin(np.abs(np.array(freqs) - float(pf))))
                            hv.update({
                                'peak_frequency': float(pf),
                                'peak_amplitude': float(pa),
                                'peak_index': idx
                            })
                            data['hv_data'] = hv
                except Exception:
                    pass
            
            return data
            
        except Exception as e:
            print(f"⚠️  Warning: Could not parse {folder.name}: {e}")
            return None
    
    def _read_velocity_model(self, model_file: Path) -> Dict:
        """Read velocity model from text file."""
        with open(model_file, 'r') as f:
            lines = f.readlines()
        
        n_layers = int(lines[0].strip())
        layers = []
        
        for i in range(1, n_layers + 1):
            parts = lines[i].strip().split()
            layers.append({
                'thickness': float(parts[0]),
                'vp': float(parts[1]),
                'vs': float(parts[2]),
                'rho': float(parts[3]),
                'is_halfspace': float(parts[0]) == 0.0
            })
        
        # Calculate derived properties
        total_thickness = sum(l['thickness'] for l in layers if not l['is_halfspace'])
        
        # Calculate impedance contrasts at interfaces
        interfaces = []
        depth = 0
        for i in range(len(layers) - 1):
            if not layers[i]['is_halfspace']:
                depth += layers[i]['thickness']
                
                # Calculate impedance contrast
                z_above = layers[i]['vs'] * layers[i]['rho']
                z_below = layers[i+1]['vs'] * layers[i+1]['rho']
                impedance_contrast = z_below / z_above if z_above > 0 else float('inf')
                
                # Calculate Vs contrast
                vs_contrast = layers[i+1]['vs'] / layers[i]['vs'] if layers[i]['vs'] > 0 else float('inf')
                
                interfaces.append({
                    'depth': depth,
                    'impedance_contrast': impedance_contrast,
                    'vs_contrast': vs_contrast,
                    'vs_above': layers[i]['vs'],
                    'vs_below': layers[i+1]['vs'],
                    'interface_index': i
                })
        
        return {
            'n_layers': n_layers,
            'layers': layers,
            'thicknesses': [l['thickness'] for l in layers],
            'vs': [l['vs'] for l in layers],
            'vp': [l['vp'] for l in layers],
            'rho': [l['rho'] for l in layers],
            'total_thickness': total_thickness,
            'interfaces': interfaces
        }
    
    def _read_hv_curve(self, csv_file: Path) -> Dict:
        """Read HV curve from CSV file."""
        try:
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            freqs = data[:, 0]
            amps = data[:, 1]
            
            # Find peak
            peak_idx = np.argmax(amps)
            
            return {
                'frequencies': freqs,
                'amplitudes': amps,
                'peak_frequency': freqs[peak_idx],
                'peak_amplitude': amps[peak_idx],
                'peak_index': peak_idx,
                'n_points': len(freqs)
            }
        except Exception as e:
            print(f"⚠️  Error reading HV curve {csv_file}: {e}")
            return {}
    
    def _read_summary(self, csv_file: Path) -> Dict:
        """Read summary CSV if available."""
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                return df.iloc[0].to_dict()
        except Exception as e:
            print(f"⚠️  Warning: Could not read summary {csv_file}: {e}")
        return {}
    
    def _analyze_data(self) -> Dict:
        """Perform comprehensive analysis of the collected data."""
        if not self.step_data:
            return {}
        
        # Extract peak frequencies and amplitudes
        peak_freqs = [step['hv_data'].get('peak_frequency', 0) for step in self.step_data]
        peak_amps = [step['hv_data'].get('peak_amplitude', 0) for step in self.step_data]
        step_numbers = [step['step'] for step in self.step_data]
        
        # Calculate frequency shifts
        initial_freq = peak_freqs[0] if peak_freqs else 1.0
        freq_shifts = [(f - initial_freq) / initial_freq * 100 for f in peak_freqs]
        
        # Find controlling interface (largest frequency change)
        max_shift_idx = np.argmax(np.abs(freq_shifts[1:])) + 1 if len(freq_shifts) > 1 else 0
        
        # Analyze impedance contrasts
        all_contrasts = []
        for step in self.step_data:
            interfaces = step['model'].get('interfaces', [])
            if interfaces:
                # Get deepest interface (most recently removed)
                deepest = interfaces[-1]
                all_contrasts.append({
                    'step': step['step'],
                    'depth': deepest['depth'],
                    'impedance_contrast': deepest['impedance_contrast'],
                    'vs_contrast': deepest['vs_contrast'],
                    'peak_frequency': step['hv_data'].get('peak_frequency', 0)
                })
        
        # Find interface with maximum impedance contrast
        max_impedance = max(all_contrasts, key=lambda x: x['impedance_contrast']) if all_contrasts else {}
        
        return {
            'initial_frequency': initial_freq,
            'final_frequency': peak_freqs[-1] if peak_freqs else 0,
            'total_frequency_shift_pct': freq_shifts[-1] if freq_shifts else 0,
            'max_frequency_shift_step': max_shift_idx,
            'max_impedance_interface': max_impedance,
            'peak_frequencies': peak_freqs,
            'peak_amplitudes': peak_amps,
            'frequency_shifts': freq_shifts,
            'step_numbers': step_numbers,
            'interface_contrasts': all_contrasts,
            'n_steps': len(self.step_data)
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Path]:
        """Generate all report components."""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*70)
        
        report_files = {}
        
        try:
            # 1. Summary CSV
            print("[1/8] Creating analysis summary CSV...")
            summary_csv = self._create_analysis_summary_csv()
            report_files['analysis_summary_csv'] = summary_csv
            
            # 2. HV Curves Overlay Plot
            print("[2/8] Creating HV curves overlay plot...")
            overlay_fig = self._create_hv_overlay_plot()
            report_files['hv_overlay_plot'] = overlay_fig
            
            # 3. Peak Evolution Analysis
            print("[3/8] Creating peak evolution analysis...")
            evolution_fig = self._create_peak_evolution_plot()
            report_files['peak_evolution_plot'] = evolution_fig
            
            # 4. Interface Analysis Plot
            print("[4/8] Creating interface analysis...")
            interface_fig = self._create_interface_analysis_plot()
            report_files['interface_analysis_plot'] = interface_fig
            
            # 5. Waterfall Plot
            print("[5/8] Creating waterfall plot...")
            waterfall_fig = self._create_waterfall_plot()
            report_files['waterfall_plot'] = waterfall_fig
            
            # 6. Publication-Ready Figure
            print("[6/8] Creating publication figure...")
            publication_fig = self._create_publication_figure()
            report_files['publication_figure'] = publication_fig
            
            # 7. Text Report
            print("[7/8] Creating text report...")
            text_report = self._create_text_report()
            report_files['text_report'] = text_report
            
            # 8. Analysis Metadata
            print("[8/8] Creating analysis metadata...")
            metadata_file = self._create_metadata()
            report_files['metadata'] = metadata_file
            
            # 9. PDF Report (multi-page with 3 steps per page)
            print("[9/9] Creating PDF report...")
            pdf_report = self._create_pdf_report()
            report_files['pdf_report'] = pdf_report
            
            print("\n[OK] REPORT GENERATION COMPLETED SUCCESSFULLY!")
            print(f"[>] All files saved in: {self.output_dir}")
            print(f"[>] Generated {len(report_files)} report components")
            
            return report_files
            
        except Exception as e:
            print(f"\n[ERROR] Error during report generation: {e}")
            import traceback
            traceback.print_exc()
            return report_files
    
    def _create_analysis_summary_csv(self) -> Path:
        """Create comprehensive analysis summary CSV."""
        csv_path = self.output_dir / 'progressive_stripping_summary.csv'
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'step', 'n_finite_layers', 'total_thickness_m', 'peak_freq_hz', 
                'peak_amplitude', 'freq_shift_pct', 'deepest_interface_depth_m',
                'deepest_vs_above', 'deepest_vs_below', 'deepest_vs_contrast',
                'deepest_impedance_contrast', 'max_impedance_contrast_in_model'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for step_data in self.step_data:
                model = step_data['model']
                hv = step_data['hv_data']
                
                # Get deepest interface
                interfaces = model.get('interfaces', [])
                deepest = interfaces[-1] if interfaces else {}
                
                # Find max impedance in this model
                max_impedance = max([iface['impedance_contrast'] for iface in interfaces]) if interfaces else 0
                
                # Calculate frequency shift
                initial_freq = self.analysis.get('initial_frequency', 1.0)
                freq_shift = (hv.get('peak_frequency', 0) - initial_freq) / initial_freq * 100
                
                writer.writerow({
                    'step': step_data['step'],
                    'n_finite_layers': step_data['n_finite_layers'],
                    'total_thickness_m': f"{model.get('total_thickness', 0):.2f}",
                    'peak_freq_hz': f"{hv.get('peak_frequency', 0):.3f}",
                    'peak_amplitude': f"{hv.get('peak_amplitude', 0):.2f}",
                    'freq_shift_pct': f"{freq_shift:.1f}",
                    'deepest_interface_depth_m': f"{deepest.get('depth', 0):.2f}",
                    'deepest_vs_above': f"{deepest.get('vs_above', 0):.0f}",
                    'deepest_vs_below': f"{deepest.get('vs_below', 0):.0f}",
                    'deepest_vs_contrast': f"{deepest.get('vs_contrast', 0):.2f}",
                    'deepest_impedance_contrast': f"{deepest.get('impedance_contrast', 0):.2f}",
                    'max_impedance_contrast_in_model': f"{max_impedance:.2f}"
                })
        
        return csv_path
    
    def _create_hv_overlay_plot(self) -> Path:
        """Create overlay plot of all HV curves."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Generate colors for each step
        n_steps = len(self.step_data)
        # Use cividis to avoid yellow tones and improve readability
        colors = plt.cm.cividis(np.linspace(0, 1, n_steps))
        
        for i, step_data in enumerate(self.step_data):
            hv = step_data['hv_data']
            freqs = hv.get('frequencies', [])
            amps = hv.get('amplitudes', [])
            
            if len(freqs) > 0 and len(amps) > 0:
                label = f"Step {step_data['step']} ({step_data['n_finite_layers']} layers)"
                ax.semilogx(freqs, amps, color=colors[i], linewidth=2, 
                           alpha=0.8, label=label)
                
                # Mark peak
                peak_f = hv.get('peak_frequency', 0)
                peak_a = hv.get('peak_amplitude', 0)
                ax.scatter(peak_f, peak_a, color=colors[i], s=60, 
                          edgecolors='white', linewidth=1, zorder=5)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12, weight='bold')
        ax.set_ylabel('H/V Amplitude Ratio', fontsize=12, weight='bold')
        ax.set_title('Progressive Layer Stripping: HV Curves Evolution', 
                    fontsize=14, weight='bold', pad=20)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Limit x-axis to >=1 Hz unless a needed peak is below 1 Hz
        min_peak = np.inf
        has_sub1_peak = False
        max_f = 0.0
        for sd in self.step_data:
            hv = sd.get('hv_data', {})
            pf = float(hv.get('peak_frequency', 0) or 0)
            if pf > 0 and pf < 1.0:
                has_sub1_peak = True
                min_peak = min(min_peak, pf)
            freqs = hv.get('frequencies', [])
            if len(freqs) > 0:
                max_f = max(max_f, float(np.max(freqs)))
        if has_sub1_peak:
            xmin = max(0.1, 0.8 * min_peak)
        else:
            xmin = 1.0
        xmax = max_f if max_f > 0 else 20.0
        ax.set_xscale('log')
        ax.set_xlim(left=xmin, right=xmax)
        # Integer ticks: 1..10 then 20,30,... or up to max
        ticks = []
        if xmax <= 12:
            ticks = list(range(max(1, int(np.ceil(xmin))), int(np.floor(xmax)) + 1))
        else:
            ticks = list(range(1, 11))
            ticks += list(range(20, int(np.floor(xmax)) + 1, 10))
        ticks = [t for t in ticks if t >= xmin and t <= xmax]
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:d}" for t in ticks])
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'hv_curves_overlay.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig_path
    
    def _create_peak_evolution_plot(self) -> Path:
        """Create peak frequency and amplitude evolution plot."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        steps = self.analysis['step_numbers']
        peak_freqs = self.analysis['peak_frequencies']
        peak_amps = self.analysis['peak_amplitudes']
        freq_shifts = self.analysis['frequency_shifts']
        
        # Panel 1: Peak frequency evolution
        ax1.plot(steps, peak_freqs, 'o-', linewidth=2, markersize=8, color='navy')
        ax1.fill_between(steps, peak_freqs, alpha=0.2, color='navy')
        ax1.set_ylabel('Peak Frequency (Hz)', fontsize=11, weight='bold')
        ax1.set_title('Peak Evolution Analysis', fontsize=12, weight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Peak amplitude evolution
        ax2.plot(steps, peak_amps, 'o-', linewidth=2, markersize=8, color='darkred')
        ax2.fill_between(steps, peak_amps, alpha=0.2, color='darkred')
        ax2.set_ylabel('Peak Amplitude', fontsize=11, weight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Frequency shift percentage
        colors = ['green' if x >= 0 else 'red' for x in freq_shifts]
        bars = ax3.bar(steps, freq_shifts, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Stripping Step', fontsize=11, weight='bold')
        ax3.set_ylabel('Frequency Shift (%)', fontsize=11, weight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, shift in zip(bars, freq_shifts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                    f'{shift:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'peak_evolution_analysis.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig_path
    
    def _create_interface_analysis_plot(self) -> Path:
        """Create interface impedance and depth analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        contrasts = self.analysis['interface_contrasts']

        if not contrasts:
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, 'No interface data available', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        else:
            # Sort by step for clearer traces
            contrasts_sorted = sorted(contrasts, key=lambda x: x['step'])
            depths = np.array([c['depth'] for c in contrasts_sorted])
            impedances = np.array([c['impedance_contrast'] for c in contrasts_sorted])
            vs_contrasts = np.array([c['vs_contrast'] for c in contrasts_sorted])
            steps = np.array([c['step'] for c in contrasts_sorted])

            # Panel 1: Impedance contrast vs depth (line + markers)
            ax1.plot(impedances, depths, '-o', color='#2E86AB', linewidth=2, markersize=5)
            ax1.invert_yaxis()
            ax1.set_xlabel('Impedance Contrast', fontsize=11, weight='bold')
            ax1.set_ylabel('Interface Depth (m)', fontsize=11, weight='bold')
            ax1.set_title('Impedance Contrast vs Depth', fontsize=12, weight='bold')
            ax1.grid(True, alpha=0.3)

            # Annotate last point
            ax1.annotate(f"Step {int(steps[-1])}", xy=(impedances[-1], depths[-1]), xytext=(5, 5), textcoords='offset points', fontsize=9)

            # Panel 2: Vs contrast vs depth (line + markers)
            ax2.plot(vs_contrasts, depths, '-o', color='#8E44AD', linewidth=2, markersize=5)
            ax2.invert_yaxis()
            ax2.set_xlabel('Vs Contrast', fontsize=11, weight='bold')
            ax2.set_ylabel('Interface Depth (m)', fontsize=11, weight='bold')
            ax2.set_title('Vs Contrast vs Depth', fontsize=12, weight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.annotate(f"Step {int(steps[-1])}", xy=(vs_contrasts[-1], depths[-1]), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'interface_analysis.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig_path
    
    def _create_waterfall_plot(self) -> Path:
        """Create waterfall plot showing progressive stripping effect."""
        fig, ax = plt.subplots(figsize=(14, 10))

        n_steps = len(self.step_data)
        # Use cividis to avoid yellow tones
        colors = plt.cm.cividis(np.linspace(0, 1, n_steps))
        
        # Normalize amplitudes and create offsets
        max_offset = 0
        for i, step_data in enumerate(self.step_data):
            hv = step_data['hv_data']
            freqs = hv.get('frequencies', [])
            amps = hv.get('amplitudes', [])
            
            if len(freqs) > 0 and len(amps) > 0:
                # Normalize to [0, 1] and add vertical offset
                max_amp = np.max(amps)
                normalized_amps = np.array(amps) / max_amp
                offset = i * 1.5  # Vertical spacing
                
                ax.semilogx(freqs, normalized_amps + offset, 
                           color=colors[i], linewidth=2, alpha=0.8,
                           label=f"Step {step_data['step']} (max: {max_amp:.1f})")
                
                # Mark peak
                peak_idx = np.argmax(amps)
                peak_f = freqs[peak_idx]
                peak_norm = normalized_amps[peak_idx] + offset
                ax.scatter(peak_f, peak_norm, color=colors[i], s=80, 
                          edgecolors='white', linewidth=2, zorder=5)
                
                max_offset = offset + 1
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12, weight='bold')
        ax.set_ylabel('Normalized H/V Amplitude (offset)', fontsize=12, weight='bold')
        ax.set_title('Progressive Layer Stripping: Waterfall View', 
                    fontsize=14, weight='bold', pad=20)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_ylim(-0.5, max_offset + 0.5)

        # Limit x-axis >=1 Hz unless needed for sub-1 Hz peaks
        min_peak = np.inf
        has_sub1_peak = False
        max_f = 0.0
        for sd in self.step_data:
            hv = sd.get('hv_data', {})
            pf = float(hv.get('peak_frequency', 0) or 0)
            if pf > 0 and pf < 1.0:
                has_sub1_peak = True
                min_peak = min(min_peak, pf)
            freqs = hv.get('frequencies', [])
            if len(freqs) > 0:
                max_f = max(max_f, float(np.max(freqs)))
        if has_sub1_peak:
            xmin = max(0.1, 0.8 * min_peak)
        else:
            xmin = 1.0
        xmax = max_f if max_f > 0 else 20.0
        ax.set_xscale('log')
        ax.set_xlim(left=xmin, right=xmax)
        # Integer ticks: 1..10 then 20,30,...
        ticks = []
        if xmax <= 12:
            ticks = list(range(max(1, int(np.ceil(xmin))), int(np.floor(xmax)) + 1))
        else:
            ticks = list(range(1, 11))
            ticks += list(range(20, int(np.floor(xmax)) + 1, 10))
        ticks = [t for t in ticks if t >= xmin and t <= xmax]
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:d}" for t in ticks])
        
        plt.tight_layout()
        
        fig_path = self.output_dir / 'waterfall_plot.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig_path
    
    def _create_publication_figure(self) -> Path:
        """Create publication-ready figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        # Panel A: Selected HV curves
        steps_to_show = [0, len(self.step_data)//2, len(self.step_data)-1]
        colors = ['blue', 'green', 'red']
        
        for i, step_idx in enumerate(steps_to_show):
            if step_idx < len(self.step_data):
                step_data = self.step_data[step_idx]
                hv = step_data['hv_data']
                freqs = hv.get('frequencies', [])
                amps = hv.get('amplitudes', [])
                
                if len(freqs) > 0:
                    ax1.semilogx(freqs, amps, color=colors[i], linewidth=2, 
                                label=f"Step {step_data['step']}")

        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('H/V Amplitude')
        ax1.set_title('(a) HVSR Evolution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Limit panel (a) x-axis >=1 Hz unless needed for sub-1 Hz peaks
        min_peak = np.inf
        has_sub1_peak = False
        max_f = 0.0
        for sd in self.step_data:
            hv = sd.get('hv_data', {})
            pf = float(hv.get('peak_frequency', 0) or 0)
            if pf > 0 and pf < 1.0:
                has_sub1_peak = True
                min_peak = min(min_peak, pf)
            freqs = hv.get('frequencies', [])
            if len(freqs) > 0:
                max_f = max(max_f, float(np.max(freqs)))
        if has_sub1_peak:
            xmin = max(0.1, 0.8 * min_peak)
        else:
            xmin = 1.0
        xmax = max_f if max_f > 0 else 20.0
        ax1.set_xscale('log')
        ax1.set_xlim(left=xmin, right=xmax)
        # Integer ticks
        ticks = []
        if xmax <= 12:
            ticks = list(range(max(1, int(np.ceil(xmin))), int(np.floor(xmax)) + 1))
        else:
            ticks = list(range(1, 11))
            ticks += list(range(20, int(np.floor(xmax)) + 1, 10))
        ticks = [t for t in ticks if t >= xmin and t <= xmax]
        if ticks:
            ax1.set_xticks(ticks)
            ax1.set_xticklabels([f"{t:d}" for t in ticks])
        
        # Panel B: Peak frequency evolution
        ax2.plot(self.analysis['step_numbers'], self.analysis['peak_frequencies'], 
                'o-', color='navy', linewidth=2, markersize=6)
        ax2.set_xlabel('Stripping Step')
        ax2.set_ylabel('Peak Frequency (Hz)')
        ax2.set_title('(b) Peak Evolution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Panel C: Impedance contrast
        contrasts = self.analysis['interface_contrasts']
        if contrasts:
            depths = [c['depth'] for c in contrasts]
            impedances = [c['impedance_contrast'] for c in contrasts]
            ax3.plot(impedances, depths, 'o-', color='darkred', linewidth=2, markersize=6)
            ax3.invert_yaxis()
        ax3.set_xlabel('Impedance Contrast')
        ax3.set_ylabel('Interface Depth (m)')
        ax3.set_title('(c) Interface Profile', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Panel D: Summary table
        ax4.axis('off')
        
        # Create summary table
        table_data = [
            ['Initial f0 (Hz)', f"{self.analysis.get('initial_frequency', 0):.3f}"],
            ['Final f0 (Hz)', f"{self.analysis.get('final_frequency', 0):.3f}"],
            ['Total shift (%)', f"{self.analysis.get('total_frequency_shift_pct', 0):.1f}"],
            ['Steps analyzed', f"{self.analysis.get('n_steps', 0)}"]
        ]
        
        table = ax4.table(cellText=table_data, colLabels=['Parameter', 'Value'],
                         cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('(d) Key Results', fontweight='bold', pad=20)
        
        plt.suptitle('Progressive Layer Stripping Analysis', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        fig_path = self.output_dir / 'publication_figure.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Also save as PDF for publications
        pdf_path = self.output_dir / 'publication_figure.pdf'
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return fig_path
    
    # ------------------------------------------------------------------ 
    # draw-on-figure variants (for interactive wizard)
    # ------------------------------------------------------------------

    def draw_hv_overlay_on_figure(self, fig, **kw) -> bool:
        """Draw HV overlay into an existing *fig*."""
        fig.clear()
        ax = fig.add_subplot(111)
        n_steps = len(self.step_data)
        if n_steps == 0:
            return False
        cmap_name = kw.get("cmap", "Blues")
        try:
            colors = plt.colormaps[cmap_name](np.linspace(0.25, 0.95, n_steps))
        except Exception:
            colors = plt.cm.Blues(np.linspace(0.25, 0.95, n_steps))
        lw = kw.get("linewidth", 2)
        alpha = kw.get("alpha", 0.8)
        log_x = kw.get("log_x", True)
        grid = kw.get("grid", True)
        font = kw.get("font_size", 12)

        show_peaks = kw.get("show_peaks", True)
        marker_size = kw.get("marker_size", 8)
        show_annot = kw.get("show_annotations", True)
        annot_size = kw.get("annotation_size", max(font - 2, 6))
        off_x = kw.get("annotation_offset_x", 6)
        off_y = kw.get("annotation_offset_y", 6)
        auto_arrange = kw.get("auto_arrange_labels", True)
        _annots = []
        for i, sd in enumerate(self.step_data):
            hv = sd['hv_data']
            freqs, amps = hv.get('frequencies', []), hv.get('amplitudes', [])
            if len(freqs) == 0:
                continue
            label = f"Step {sd['step']} ({sd['n_finite_layers']}L)"
            ax.plot(freqs, amps, color=colors[i], linewidth=lw,
                    alpha=alpha, label=label)
            if show_peaks:
                pf = hv.get('peak_frequency', 0)
                pa = hv.get('peak_amplitude', 0)
                if pf:
                    ax.scatter(pf, pa, color=colors[i], s=marker_size * 10,
                               edgecolors='white', linewidth=1, zorder=5)
                    if show_annot:
                        ann = ax.annotate(
                            f"{pf:.2f} Hz\n{pa:.1f}",
                            (pf, pa), fontsize=annot_size,
                            xytext=(off_x, off_y),
                            textcoords='offset points',
                            ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc='white', alpha=0.7, lw=0.5))
                        _annots.append(ann)
        if auto_arrange and _annots:
            _spread_annotations(_annots)

        if log_x:
            ax.set_xscale('log')
        if grid:
            ax.grid(True, alpha=0.3, which='both')
        else:
            ax.grid(False)
        ax.set_xlabel('Frequency (Hz)', fontsize=font, weight='bold')
        ax.set_ylabel('H/V Amplitude Ratio', fontsize=font, weight='bold')
        ax.set_title('Progressive Layer Stripping: HV Curves Evolution',
                      fontsize=font + 2, weight='bold')
        ax.legend(fontsize=max(font - 2, 6), loc='upper right')
        xlim_min = kw.get("xlim_min")
        xlim_max = kw.get("xlim_max")
        if xlim_min is not None:
            ax.set_xlim(left=xlim_min)
        if xlim_max is not None:
            ax.set_xlim(right=xlim_max)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return True

    def draw_peak_evolution_on_figure(self, fig, **kw) -> bool:
        """Draw 3-panel peak evolution into *fig*."""
        fig.clear()
        if not self.analysis.get('step_numbers'):
            return False
        steps = self.analysis['step_numbers']
        peak_freqs = self.analysis['peak_frequencies']
        peak_amps = self.analysis['peak_amplitudes']
        freq_shifts = self.analysis['frequency_shifts']
        font = kw.get("font_size", 11)
        grid = kw.get("grid", True)
        show_fill = kw.get("show_fill", True)
        ms = kw.get("marker_size", 8)
        lw = kw.get("linewidth", 2)
        show_annot = kw.get("show_annotations", True)
        annot_size = kw.get("annotation_size", max(font - 2, 6))

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(steps, peak_freqs, 'o-', lw=lw, ms=ms, color='navy')
        if show_fill:
            ax1.fill_between(steps, peak_freqs, alpha=0.2, color='navy')
        if show_annot:
            for s, pf in zip(steps, peak_freqs):
                ax1.annotate(f"{pf:.2f}", (s, pf), fontsize=annot_size,
                             xytext=(3, 5), textcoords='offset points')
        ax1.set_ylabel('Peak Freq (Hz)', fontsize=font, weight='bold')
        ax1.set_title('Peak Evolution', fontsize=font + 1, weight='bold')
        if grid:
            ax1.grid(True, alpha=0.3)
        else:
            ax1.grid(False)

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(steps, peak_amps, 'o-', lw=lw, ms=ms, color='darkred')
        if show_fill:
            ax2.fill_between(steps, peak_amps, alpha=0.2, color='darkred')
        if show_annot:
            for s, pa in zip(steps, peak_amps):
                ax2.annotate(f"{pa:.1f}", (s, pa), fontsize=annot_size,
                             xytext=(3, 5), textcoords='offset points')
        ax2.set_ylabel('Peak Amplitude', fontsize=font, weight='bold')
        if grid:
            ax2.grid(True, alpha=0.3)
        else:
            ax2.grid(False)

        ax3 = fig.add_subplot(3, 1, 3)
        colors_bar = ['green' if x >= 0 else 'red' for x in freq_shifts]
        ax3.bar(steps, freq_shifts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', lw=1)
        ax3.set_xlabel('Stripping Step', fontsize=font, weight='bold')
        ax3.set_ylabel('Freq Shift (%)', fontsize=font, weight='bold')
        if grid:
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.grid(False)

        try:
            fig.tight_layout()
        except Exception:
            pass
        return True

    def draw_interface_analysis_on_figure(self, fig, **kw) -> bool:
        """Draw interface impedance / Vs contrast into *fig*."""
        fig.clear()
        contrasts = self.analysis.get('interface_contrasts', [])
        font = kw.get("font_size", 11)
        grid = kw.get("grid", True)
        show_annot = kw.get("show_annotations", True)
        off_x = kw.get("annotation_offset_x", 5)
        off_y = kw.get("annotation_offset_y", 0)
        auto_arrange = kw.get("auto_arrange_labels", True)
        ms = kw.get("marker_size", 8)
        lw = kw.get("linewidth", 2)
        af = kw.get("annot_font", font)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        if not contrasts:
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, 'No interface data', ha='center',
                        va='center', transform=ax.transAxes)
        else:
            cs = sorted(contrasts, key=lambda x: x['step'])
            depths = [c['depth'] for c in cs]
            imps = [c['impedance_contrast'] for c in cs]
            vsc = [c['vs_contrast'] for c in cs]

            ax1.plot(imps, depths, '-o', color='#2E86AB', lw=lw, ms=ms)
            ax1.invert_yaxis()
            ax1.set_xlabel('Impedance Contrast', fontsize=font, weight='bold')
            ax1.set_ylabel('Depth (m)', fontsize=font, weight='bold')
            ax1.set_title('Impedance vs Depth', fontsize=font + 1, weight='bold')
            if show_annot:
                _annots_i1 = []
                for d, imp in zip(depths, imps):
                    ann = ax1.annotate(f"{imp:.2f}", (imp, d), fontsize=af,
                                 xytext=(off_x, off_y),
                                 textcoords='offset points')
                    _annots_i1.append(ann)
                if auto_arrange:
                    _spread_annotations(_annots_i1)

            ax2.plot(vsc, depths, '-o', color='#8E44AD', lw=lw, ms=ms)
            ax2.invert_yaxis()
            ax2.set_xlabel('Vs Contrast', fontsize=font, weight='bold')
            ax2.set_ylabel('Depth (m)', fontsize=font, weight='bold')
            ax2.set_title('Vs Contrast vs Depth', fontsize=font + 1, weight='bold')
            if show_annot:
                _annots_i2 = []
                for d, v in zip(depths, vsc):
                    ann = ax2.annotate(f"{v:.2f}", (v, d), fontsize=af,
                                 xytext=(off_x, off_y),
                                 textcoords='offset points')
                    _annots_i2.append(ann)
                if auto_arrange:
                    _spread_annotations(_annots_i2)

        for ax in (ax1, ax2):
            if grid:
                ax.grid(True, alpha=0.3)
            else:
                ax.grid(False)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return True

    def draw_waterfall_on_figure(self, fig, **kw) -> bool:
        """Draw waterfall plot into *fig*."""
        fig.clear()
        ax = fig.add_subplot(111)
        n_steps = len(self.step_data)
        if n_steps == 0:
            return False
        cmap_name = kw.get("cmap", "Blues")
        try:
            colors = plt.colormaps[cmap_name](np.linspace(0.25, 0.95, n_steps))
        except Exception:
            colors = plt.cm.Blues(np.linspace(0.25, 0.95, n_steps))
        offset_factor = kw.get("offset_factor", 1.5)
        lw = kw.get("linewidth", 2)
        log_x = kw.get("log_x", True)
        grid = kw.get("grid", True)
        font = kw.get("font_size", 12)

        normalize = kw.get("normalize", False)
        alpha = kw.get("alpha", 0.8)
        show_annot = kw.get("show_annotations", True)
        annot_size = kw.get("annotation_size", max(font - 2, 6))
        max_offset = 0
        for i, sd in enumerate(self.step_data):
            hv = sd['hv_data']
            freqs, amps = hv.get('frequencies', []), hv.get('amplitudes', [])
            if len(freqs) == 0:
                continue
            max_amp = np.max(amps)
            if normalize or True:
                plot_amps = np.array(amps) / max_amp
            else:
                plot_amps = np.array(amps)
            offset = i * offset_factor
            ax.plot(freqs, plot_amps + offset, color=colors[i], lw=lw, alpha=alpha,
                    label=f"Step {sd['step']} (max: {max_amp:.1f})")
            peak_idx = np.argmax(amps)
            peak_x = freqs[peak_idx]
            peak_y = plot_amps[peak_idx] + offset
            ax.scatter(peak_x, peak_y,
                       color=colors[i], s=80, edgecolors='white', lw=2, zorder=5)
            if show_annot:
                ax.annotate(
                    f"{peak_x:.2f} Hz",
                    (peak_x, peak_y), fontsize=annot_size,
                    xytext=(5, 5), textcoords='offset points',
                    ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15',
                              fc='white', alpha=0.7, lw=0.5))
            max_offset = offset + 1

        if log_x:
            ax.set_xscale('log')
        if grid:
            ax.grid(True, alpha=0.3, which='both')
        else:
            ax.grid(False)
        ax.set_xlabel('Frequency (Hz)', fontsize=font, weight='bold')
        ax.set_ylabel('Normalized H/V (offset)', fontsize=font, weight='bold')
        ax.set_title('Waterfall View', fontsize=font + 2, weight='bold')
        ax.legend(fontsize=max(font - 3, 6), loc='upper right')
        ax.set_ylim(-0.5, max_offset + 0.5)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return True

    def draw_publication_on_figure(self, fig, **kw) -> bool:
        """Draw 2x2 publication figure into *fig*."""
        fig.clear()
        if not self.step_data:
            return False
        font = kw.get("font_size", 10)
        grid = kw.get("grid", True)
        cmap_name = kw.get("cmap", "Blues")
        lw = kw.get("linewidth", 2)
        alpha = kw.get("alpha", 0.85)
        show_annot = kw.get("show_annotations", True)
        annot_size = kw.get("annotation_size", max(font - 1, 6))
        off_x = kw.get("annotation_offset_x", 4)
        off_y = kw.get("annotation_offset_y", 4)
        auto_arrange = kw.get("auto_arrange_labels", True)

        n_steps = len(self.step_data)
        try:
            cmap_colors = plt.colormaps[cmap_name](
                np.linspace(0.25, 0.95, n_steps))
        except Exception:
            cmap_colors = plt.cm.Blues(np.linspace(0.25, 0.95, n_steps))

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        # (a) HV curves — first, mid, last
        _annots_a = []
        indices = [0, len(self.step_data) // 2, len(self.step_data) - 1]
        for ci, si in enumerate(indices):
            if si < len(self.step_data):
                sd = self.step_data[si]
                hv = sd['hv_data']
                f, a = hv.get('frequencies', []), hv.get('amplitudes', [])
                if len(f):
                    ax1.semilogx(f, a, color=cmap_colors[si], lw=lw,
                                 alpha=alpha,
                                 label=f"Step {sd['step']}")
                    if show_annot:
                        pf = hv.get('peak_frequency', 0)
                        pa = hv.get('peak_amplitude', 0)
                        if pf:
                            ax1.scatter(pf, pa, color=cmap_colors[si],
                                        s=50, edgecolors='white', lw=1,
                                        zorder=5)
                            ann = ax1.annotate(
                                f"{pf:.2f}", (pf, pa), fontsize=annot_size,
                                xytext=(off_x, off_y),
                                textcoords='offset points')
                            _annots_a.append(ann)
        if auto_arrange and _annots_a:
            _spread_annotations(_annots_a)
        ax1.set_xlabel('Frequency (Hz)', fontsize=font)
        ax1.set_ylabel('H/V Amplitude', fontsize=font)
        ax1.set_title('(a) HVSR Evolution', fontweight='bold', fontsize=font)
        if grid:
            ax1.grid(True, alpha=0.3)
        else:
            ax1.grid(False)
        ax1.legend(fontsize=max(font - 2, 6))

        # (b) Peak frequency
        step_nums = self.analysis['step_numbers']
        peak_freqs = self.analysis['peak_frequencies']
        ax2.plot(step_nums, peak_freqs, 'o-', color=cmap_colors[n_steps // 2],
                 lw=lw, ms=6, alpha=alpha)
        if show_annot:
            _annots_b = []
            for s, pf in zip(step_nums, peak_freqs):
                ann = ax2.annotate(f"{pf:.2f}", (s, pf), fontsize=annot_size,
                             xytext=(off_x, off_y),
                             textcoords='offset points')
                _annots_b.append(ann)
            if auto_arrange:
                _spread_annotations(_annots_b)
        ax2.set_xlabel('Step', fontsize=font)
        ax2.set_ylabel('Peak Freq (Hz)', fontsize=font)
        ax2.set_title('(b) Peak Evolution', fontweight='bold', fontsize=font)
        if grid:
            ax2.grid(True, alpha=0.3)
        else:
            ax2.grid(False)

        # (c) Impedance
        contrasts = self.analysis.get('interface_contrasts', [])
        if contrasts:
            depths = [c['depth'] for c in contrasts]
            imps = [c['impedance_contrast'] for c in contrasts]
            ax3.plot(imps, depths, 'o-', color=cmap_colors[-1], lw=lw,
                     ms=6, alpha=alpha)
            ax3.invert_yaxis()
            if show_annot:
                _annots_c = []
                for d, imp in zip(depths, imps):
                    ann = ax3.annotate(f"{imp:.2f}", (imp, d),
                                 fontsize=annot_size,
                                 xytext=(off_x, 0),
                                 textcoords='offset points')
                    _annots_c.append(ann)
                if auto_arrange:
                    _spread_annotations(_annots_c)
        ax3.set_xlabel('Impedance Contrast', fontsize=font)
        ax3.set_ylabel('Depth (m)', fontsize=font)
        ax3.set_title('(c) Interface Profile', fontweight='bold', fontsize=font)
        if grid:
            ax3.grid(True, alpha=0.3)
        else:
            ax3.grid(False)

        # (d) Summary table
        ax4.axis('off')
        table_data = [
            ['Initial f0', f"{self.analysis.get('initial_frequency', 0):.3f} Hz"],
            ['Final f0', f"{self.analysis.get('final_frequency', 0):.3f} Hz"],
            ['Total shift', f"{self.analysis.get('total_frequency_shift_pct', 0):.1f}%"],
            ['Steps', f"{self.analysis.get('n_steps', 0)}"],
        ]
        tbl = ax4.table(cellText=table_data, colLabels=['Param', 'Value'],
                        cellLoc='left', loc='center', colWidths=[0.55, 0.45])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(kw.get("table_font", font))
        tbl.scale(1, 1.8)
        ax4.set_title('(d) Key Results', fontweight='bold', fontsize=font)

        fig.suptitle('Progressive Layer Stripping Analysis',
                     fontsize=font + 2, fontweight='bold', y=0.99)
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception:
            pass
        return True

    def _create_text_report(self) -> Path:
        """Create comprehensive text report."""
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("PROGRESSIVE LAYER STRIPPING ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis directory: {self.strip_dir}\n")
            f.write(f"Number of steps analyzed: {len(self.step_data)}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*30 + "\n")
            analysis = self.analysis
            f.write(f"Initial peak frequency: {analysis.get('initial_frequency', 0):.3f} Hz\n")
            f.write(f"Final peak frequency: {analysis.get('final_frequency', 0):.3f} Hz\n")
            f.write(f"Total frequency shift: {analysis.get('total_frequency_shift_pct', 0):.1f}%\n")
            
            if analysis.get('max_impedance_interface'):
                max_imp = analysis['max_impedance_interface']
                f.write(f"Maximum impedance contrast: {max_imp.get('impedance_contrast', 0):.2f} ")
                f.write(f"at {max_imp.get('depth', 0):.1f} m depth\n")
            f.write("\n")
            
            # Detailed Step Analysis
            f.write("DETAILED STEP ANALYSIS\n")
            f.write("-"*30 + "\n")
            f.write(f"{'Step':<5} {'Layers':<7} {'Depth(m)':<8} {'f0(Hz)':<8} {'Amplitude':<10} {'Shift(%)':<8}\n")
            f.write("-"*50 + "\n")
            
            for i, step_data in enumerate(self.step_data):
                model = step_data['model']
                hv = step_data['hv_data']
                interfaces = model.get('interfaces', [])
                deepest_depth = interfaces[-1]['depth'] if interfaces else 0
                freq_shift = analysis['frequency_shifts'][i] if i < len(analysis['frequency_shifts']) else 0
                
                f.write(f"{step_data['step']:<5} {step_data['n_finite_layers']:<7} ")
                f.write(f"{deepest_depth:<8.1f} {hv.get('peak_frequency', 0):<8.3f} ")
                f.write(f"{hv.get('peak_amplitude', 0):<10.2f} {freq_shift:<8.1f}\n")
            
            f.write("\n")
            
            # Conclusions
            f.write("CONCLUSIONS\n")
            f.write("-"*30 + "\n")
            if analysis.get('total_frequency_shift_pct', 0) > 5:
                f.write("• Significant frequency shift observed during stripping\n")
            else:
                f.write("• Minimal frequency shift observed during stripping\n")
            
            contrasts = analysis.get('interface_contrasts', [])
            if contrasts:
                max_contrast = max(contrasts, key=lambda x: x['impedance_contrast'])
                f.write(f"• Maximum impedance contrast of {max_contrast['impedance_contrast']:.2f} ")
                f.write(f"at {max_contrast['depth']:.1f} m depth\n")
            
            f.write("• Analysis completed successfully\n")
        
        return report_path
    
    def _create_metadata(self) -> Path:
        """Create analysis metadata JSON file."""
        metadata_path = self.output_dir / 'analysis_metadata.json'
        
        metadata = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'strip_directory': str(self.strip_dir),
                'output_directory': str(self.output_dir),
                'n_steps_analyzed': len(self.step_data),
                'analysis_type': 'progressive_layer_stripping'
            },
            'summary_statistics': self.analysis,
            'step_details': []
        }
        
        # Add step details
        for step_data in self.step_data:
            step_info = {
                'step': step_data['step'],
                'n_finite_layers': step_data['n_finite_layers'],
                'folder_name': step_data['name'],
                'peak_frequency': step_data['hv_data'].get('peak_frequency', 0),
                'peak_amplitude': step_data['hv_data'].get('peak_amplitude', 0),
                'total_thickness': step_data['model'].get('total_thickness', 0),
                'n_interfaces': len(step_data['model'].get('interfaces', []))
            }
            metadata['step_details'].append(step_info)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return metadata_path
    
    def _create_pdf_report(self) -> Path:
        """Create a multi-page PDF report with 3 steps per page.
        
        Each step shows:
        - Vs profile with depth annotation
        - HVSR curve with selected peak
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        pdf_path = self.output_dir / 'progressive_stripping_report.pdf'
        
        # Calculate number of pages needed (3 steps per page)
        steps_per_page = 3
        n_pages = (len(self.step_data) + steps_per_page - 1) // steps_per_page
        
        with PdfPages(pdf_path) as pdf:
            # Title page
            fig_title = plt.figure(figsize=(11, 8.5))
            fig_title.text(0.5, 0.6, 'Progressive Layer Stripping Analysis', 
                          ha='center', va='center', fontsize=24, fontweight='bold')
            fig_title.text(0.5, 0.45, f'Total Steps: {len(self.step_data)}', 
                          ha='center', va='center', fontsize=16)
            fig_title.text(0.5, 0.35, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                          ha='center', va='center', fontsize=12, style='italic')
            fig_title.text(0.5, 0.25, f'Source: {self.strip_dir}', 
                          ha='center', va='center', fontsize=10, color='gray')
            pdf.savefig(fig_title)
            plt.close(fig_title)
            
            # Create pages with 3 steps each
            for page_idx in range(n_pages):
                start_idx = page_idx * steps_per_page
                end_idx = min(start_idx + steps_per_page, len(self.step_data))
                page_steps = self.step_data[start_idx:end_idx]
                
                # Create figure with grid for this page
                fig = plt.figure(figsize=(11, 8.5))
                
                for row_idx, step_data in enumerate(page_steps):
                    self._add_step_to_pdf_page(fig, step_data, row_idx, len(page_steps))
                
                # Add page number
                fig.text(0.5, 0.02, f'Page {page_idx + 2} of {n_pages + 1}', 
                        ha='center', fontsize=9, color='gray')
                
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                pdf.savefig(fig)
                plt.close(fig)
        
        return pdf_path
    
    def _add_step_to_pdf_page(self, fig, step_data: Dict, row_idx: int, n_rows: int):
        """Add a single step's plots to a PDF page row."""
        # Calculate subplot positions (2 columns per row)
        n_cols = 2
        
        # Calculate row height and position
        row_height = 0.9 / n_rows
        row_bottom = 0.9 - (row_idx + 1) * row_height + 0.05
        
        # Left subplot: HVSR Curve (wider)
        ax_hv = fig.add_axes([0.08, row_bottom, 0.50, row_height - 0.08])
        # Right subplot: Vs Profile (narrower)
        ax_vs = fig.add_axes([0.62, row_bottom, 0.28, row_height - 0.08])
        
        step_name = step_data['name']
        model_info = step_data.get('model', {})
        hv_info = step_data.get('hv_data', {})
        
        # === Plot HVSR Curve (LEFT) ===
        freqs = hv_info.get('frequencies', np.array([]))
        amps = hv_info.get('amplitudes', np.array([]))
        peak_freq = hv_info.get('peak_frequency', 0)
        peak_amp = hv_info.get('peak_amplitude', 0)
        
        if len(freqs) > 0 and len(amps) > 0:
            ax_hv.semilogx(freqs, amps, 'b-', linewidth=1.5, label='H/V')
            
            # Mark peak
            if peak_freq > 0:
                ax_hv.scatter(peak_freq, peak_amp, color='red', s=100, marker='*',
                           edgecolors='darkred', linewidth=1, zorder=5)
                ax_hv.axvline(x=peak_freq, color='red', linestyle='--', alpha=0.5)
                ax_hv.text(0.95, 0.95, f'f0 = {peak_freq:.2f} Hz', 
                        transform=ax_hv.transAxes, fontsize=9, fontweight='bold',
                        ha='right', va='top', color='red',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax_hv.set_xlim(freqs[0], freqs[-1])
        
        ax_hv.set_xlabel('Frequency (Hz)', fontsize=9)
        ax_hv.set_ylabel('H/V Amplitude', fontsize=9)
        ax_hv.set_title(f'{step_name} - HVSR Curve', fontsize=10, fontweight='bold')
        ax_hv.grid(True, alpha=0.3, which='both')
        
        # === Plot Vs Profile (RIGHT) ===
        thicknesses = model_info.get('thicknesses', [])
        vs_values = model_info.get('vs', [])
        
        if thicknesses and vs_values:
            # Calculate depths
            depths = [0]
            total_finite = sum(h for h in thicknesses if h > 0)
            for h in thicknesses:
                if h > 0:
                    depths.append(depths[-1] + h)
                else:
                    depths.append(depths[-1] + compute_halfspace_display_depth(total_finite))
            
            # Create step profile
            plot_depths = []
            plot_vs = []
            for i in range(len(vs_values)):
                plot_depths.extend([depths[i], depths[i + 1]])
                plot_vs.extend([vs_values[i], vs_values[i]])
            
            # Plot
            ax_vs.fill_betweenx(plot_depths, 0, plot_vs, alpha=0.3, color='teal')
            ax_vs.step(plot_vs + [plot_vs[-1]], [0] + plot_depths, where='pre', 
                      color='teal', linewidth=1.5, linestyle='-')
            
            # Highlight deepest interface
            if len(depths) > 2:
                deepest_depth = depths[-2]
                ax_vs.axhline(y=deepest_depth, color='red', linestyle='-', alpha=0.6, linewidth=1.5)
                # Compact annotation
                ax_vs.text(0.95, 0.02, f'{deepest_depth:.0f}m', 
                          transform=ax_vs.transAxes, fontsize=8, fontweight='bold',
                          color='red', ha='right', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax_vs.set_xlim(0, max(plot_vs) * 1.2 if plot_vs else 1000)
            ax_vs.invert_yaxis()
        
        ax_vs.set_xlabel('Vs (m/s)', fontsize=8)
        ax_vs.set_ylabel('Depth (m)', fontsize=8)
        ax_vs.set_title(f'{step_name} - Vs Profile', fontsize=9, fontweight='bold')
        ax_vs.grid(True, alpha=0.3)
        ax_vs.tick_params(axis='both', labelsize=7)


__all__ = [
    "ProgressiveStrippingReporter"
]
