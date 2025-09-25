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
        
        print(f"📊 Initialized reporter with {len(self.step_data)} steps")
        print(f"📁 Output directory: {self.output_dir}")
    
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
        print("📊 GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("="*70)
        
        report_files = {}
        
        try:
            # 1. Summary CSV
            print("📝 Creating analysis summary CSV...")
            summary_csv = self._create_analysis_summary_csv()
            report_files['analysis_summary_csv'] = summary_csv
            
            # 2. HV Curves Overlay Plot
            print("📈 Creating HV curves overlay plot...")
            overlay_fig = self._create_hv_overlay_plot()
            report_files['hv_overlay_plot'] = overlay_fig
            
            # 3. Peak Evolution Analysis
            print("📊 Creating peak evolution analysis...")
            evolution_fig = self._create_peak_evolution_plot()
            report_files['peak_evolution_plot'] = evolution_fig
            
            # 4. Interface Analysis Plot
            print("🔍 Creating interface analysis...")
            interface_fig = self._create_interface_analysis_plot()
            report_files['interface_analysis_plot'] = interface_fig
            
            # 5. Waterfall Plot
            print("🌊 Creating waterfall plot...")
            waterfall_fig = self._create_waterfall_plot()
            report_files['waterfall_plot'] = waterfall_fig
            
            # 6. Publication-Ready Figure
            print("📚 Creating publication figure...")
            publication_fig = self._create_publication_figure()
            report_files['publication_figure'] = publication_fig
            
            # 7. Text Report
            print("📄 Creating text report...")
            text_report = self._create_text_report()
            report_files['text_report'] = text_report
            
            # 8. Analysis Metadata
            print("📋 Creating analysis metadata...")
            metadata_file = self._create_metadata()
            report_files['metadata'] = metadata_file
            
            print("\n✅ REPORT GENERATION COMPLETED SUCCESSFULLY!")
            print(f"📁 All files saved in: {self.output_dir}")
            print(f"📊 Generated {len(report_files)} report components")
            
            return report_files
            
        except Exception as e:
            print(f"\n❌ Error during report generation: {e}")
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


__all__ = [
    "ProgressiveStrippingReporter"
]
