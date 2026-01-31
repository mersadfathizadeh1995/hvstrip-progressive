"""
Advanced Analysis Module for HVSR Progressive Layer Stripping.

This module provides:
1. Statistical analysis of stripping results
2. Controlling interface detection algorithms
3. Layer contribution analysis
4. Comparative analysis across stripping steps
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.signal import find_peaks
from scipy.stats import pearsonr


class StrippingAnalyzer:
    """Analyzer for progressive layer stripping results."""

    def __init__(self, strip_directory: Path):
        """
        Initialize analyzer with strip directory.

        Parameters:
            strip_directory: Path to strip directory containing Step folders
        """
        self.strip_dir = Path(strip_directory)
        self.steps_data = []
        self.load_all_steps()

    def load_all_steps(self):
        """Load H/V data from all stripping steps."""
        step_folders = sorted([d for d in self.strip_dir.iterdir()
                              if d.is_dir() and d.name.startswith('Step')])

        for step_folder in step_folders:
            step_name = step_folder.name
            hv_csv = step_folder / "hv_curve.csv"

            if not hv_csv.exists():
                continue

            # Parse step info
            parts = step_name.split('_')
            step_num = int(parts[0].replace('Step', ''))
            n_layers = int(parts[1].split('-')[0])

            # Load H/V data
            data = np.loadtxt(hv_csv, delimiter=',', skiprows=1)
            freqs = data[:, 0]
            amps = data[:, 1]

            # Find peak
            peak_idx = np.argmax(amps)
            peak_freq = freqs[peak_idx]
            peak_amp = amps[peak_idx]

            self.steps_data.append({
                'step': step_num,
                'n_layers': n_layers,
                'freqs': freqs,
                'amps': amps,
                'peak_freq': peak_freq,
                'peak_amp': peak_amp,
                'step_name': step_name,
                'folder': step_folder
            })

        print(f"Loaded {len(self.steps_data)} stripping steps")

    def compute_statistics(self) -> Dict:
        """
        Compute comprehensive statistics across all stripping steps.

        Returns:
            Dictionary containing statistical metrics
        """
        if not self.steps_data:
            return {}

        stats = {
            'n_steps': len(self.steps_data),
            'layer_counts': [s['n_layers'] for s in self.steps_data],
            'peak_frequencies': [s['peak_freq'] for s in self.steps_data],
            'peak_amplitudes': [s['peak_amp'] for s in self.steps_data],
        }

        # Peak frequency statistics
        peak_freqs = np.array(stats['peak_frequencies'])
        stats['peak_freq_mean'] = np.mean(peak_freqs)
        stats['peak_freq_std'] = np.std(peak_freqs)
        stats['peak_freq_range'] = (np.min(peak_freqs), np.max(peak_freqs))
        stats['peak_freq_change_total'] = peak_freqs[-1] - peak_freqs[0] if len(peak_freqs) > 1 else 0

        # Peak amplitude statistics
        peak_amps = np.array(stats['peak_amplitudes'])
        stats['peak_amp_mean'] = np.mean(peak_amps)
        stats['peak_amp_std'] = np.std(peak_amps)
        stats['peak_amp_range'] = (np.min(peak_amps), np.max(peak_amps))
        stats['peak_amp_change_total'] = peak_amps[-1] - peak_amps[0] if len(peak_amps) > 1 else 0

        # Step-wise changes
        if len(self.steps_data) > 1:
            freq_changes = np.diff(peak_freqs)
            amp_changes = np.diff(peak_amps)

            stats['peak_freq_changes'] = freq_changes
            stats['peak_amp_changes'] = amp_changes
            stats['max_freq_change_step'] = int(np.argmax(np.abs(freq_changes)))
            stats['max_amp_change_step'] = int(np.argmax(np.abs(amp_changes)))

        return stats

    def detect_controlling_interfaces(self, threshold_percentile: float = 75) -> List[Dict]:
        """
        Detect controlling interfaces based on significant H/V changes.

        A controlling interface is identified when removing a layer causes
        a significant change in the H/V peak characteristics.

        Parameters:
            threshold_percentile: Percentile threshold for "significant" change

        Returns:
            List of dictionaries describing controlling interfaces
        """
        if len(self.steps_data) < 2:
            return []

        controlling_interfaces = []

        # Calculate step-wise changes
        for i in range(len(self.steps_data) - 1):
            curr_step = self.steps_data[i]
            next_step = self.steps_data[i + 1]

            # Changes in peak characteristics
            freq_change = abs(next_step['peak_freq'] - curr_step['peak_freq'])
            freq_change_pct = (freq_change / curr_step['peak_freq']) * 100

            amp_change = abs(next_step['peak_amp'] - curr_step['peak_amp'])
            amp_change_pct = (amp_change / curr_step['peak_amp']) * 100

            # Correlation between H/V curves
            # Interpolate to common frequency grid for comparison
            common_freqs = curr_step['freqs']
            corr_amp = np.interp(common_freqs, next_step['freqs'], next_step['amps'])
            correlation, _ = pearsonr(curr_step['amps'], corr_amp)

            # Calculate significance score
            # High score = significant change = controlling interface
            significance_score = (
                freq_change_pct * 0.4 +  # Weight frequency change
                amp_change_pct * 0.3 +    # Weight amplitude change
                (1 - correlation) * 100 * 0.3  # Weight curve dissimilarity
            )

            controlling_interfaces.append({
                'step': i,
                'from_layers': curr_step['n_layers'],
                'to_layers': next_step['n_layers'],
                'removed_layer_depth': curr_step['n_layers'],  # Approximate
                'freq_change': freq_change,
                'freq_change_pct': freq_change_pct,
                'amp_change': amp_change,
                'amp_change_pct': amp_change_pct,
                'correlation': correlation,
                'significance_score': significance_score
            })

        # Sort by significance
        controlling_interfaces.sort(key=lambda x: x['significance_score'], reverse=True)

        # Mark significant ones based on threshold
        if controlling_interfaces:
            scores = [ci['significance_score'] for ci in controlling_interfaces]
            threshold = np.percentile(scores, threshold_percentile)

            for ci in controlling_interfaces:
                ci['is_controlling'] = ci['significance_score'] >= threshold

        return controlling_interfaces

    def analyze_layer_contributions(self) -> pd.DataFrame:
        """
        Analyze the contribution of each layer to the overall H/V curve.

        Returns:
            DataFrame with layer-by-layer contribution analysis
        """
        if len(self.steps_data) < 2:
            return pd.DataFrame()

        contributions = []

        for i in range(len(self.steps_data) - 1):
            curr_step = self.steps_data[i]
            next_step = self.steps_data[i + 1]

            # Layer being removed (approximate as deepest finite layer)
            removed_layer_num = curr_step['n_layers']

            # Impact metrics
            freq_impact = next_step['peak_freq'] - curr_step['peak_freq']
            amp_impact = next_step['peak_amp'] - curr_step['peak_amp']

            # Spectral energy change (integral of squared difference)
            common_freqs = curr_step['freqs']
            interp_amps = np.interp(common_freqs, next_step['freqs'], next_step['amps'])
            spectral_diff = np.trapz((curr_step['amps'] - interp_amps)**2, common_freqs)

            contributions.append({
                'step': i,
                'removed_layer': removed_layer_num,
                'from_n_layers': curr_step['n_layers'],
                'to_n_layers': next_step['n_layers'],
                'freq_shift_hz': freq_impact,
                'amp_change': amp_impact,
                'spectral_energy_change': spectral_diff,
                'original_peak_freq': curr_step['peak_freq'],
                'new_peak_freq': next_step['peak_freq'],
                'original_peak_amp': curr_step['peak_amp'],
                'new_peak_amp': next_step['peak_amp']
            })

        df = pd.DataFrame(contributions)
        return df

    def generate_analysis_report(self, output_path: Path):
        """
        Generate comprehensive analysis report.

        Parameters:
            output_path: Path for output report file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HVSR PROGRESSIVE LAYER STRIPPING - ADVANCED ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            # Basic statistics
            stats = self.compute_statistics()
            f.write("1. STATISTICAL SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total stripping steps: {stats['n_steps']}\n")
            f.write(f"Layer counts: {stats['layer_counts'][0]} → {stats['layer_counts'][-1]}\n\n")

            f.write(f"Peak Frequency Analysis:\n")
            f.write(f"  Range: {stats['peak_freq_range'][0]:.2f} - {stats['peak_freq_range'][1]:.2f} Hz\n")
            f.write(f"  Mean: {stats['peak_freq_mean']:.2f} Hz\n")
            f.write(f"  Std Dev: {stats['peak_freq_std']:.2f} Hz\n")
            f.write(f"  Total Change: {stats['peak_freq_change_total']:.2f} Hz\n\n")

            f.write(f"Peak Amplitude Analysis:\n")
            f.write(f"  Range: {stats['peak_amp_range'][0]:.2f} - {stats['peak_amp_range'][1]:.2f}\n")
            f.write(f"  Mean: {stats['peak_amp_mean']:.2f}\n")
            f.write(f"  Std Dev: {stats['peak_amp_std']:.2f}\n")
            f.write(f"  Total Change: {stats['peak_amp_change_total']:.2f}\n\n")

            # Controlling interfaces
            f.write("\n2. CONTROLLING INTERFACE DETECTION\n")
            f.write("-"*80 + "\n")

            controlling = self.detect_controlling_interfaces()

            if controlling:
                f.write(f"Identified {sum(1 for c in controlling if c.get('is_controlling', False))} "
                       f"controlling interfaces:\n\n")

                for ci in controlling:
                    if ci.get('is_controlling', False):
                        f.write(f"Step {ci['step']} → {ci['step']+1}: "
                               f"{ci['from_layers']}-layer → {ci['to_layers']}-layer\n")
                        f.write(f"  Significance Score: {ci['significance_score']:.2f}\n")
                        f.write(f"  Frequency Change: {ci['freq_change']:.3f} Hz "
                               f"({ci['freq_change_pct']:.1f}%)\n")
                        f.write(f"  Amplitude Change: {ci['amp_change']:.3f} "
                               f"({ci['amp_change_pct']:.1f}%)\n")
                        f.write(f"  Curve Correlation: {ci['correlation']:.3f}\n\n")
            else:
                f.write("No significant controlling interfaces detected.\n\n")

            # Layer contributions
            f.write("\n3. LAYER-BY-LAYER CONTRIBUTION ANALYSIS\n")
            f.write("-"*80 + "\n")

            contributions_df = self.analyze_layer_contributions()

            if not contributions_df.empty:
                for _, row in contributions_df.iterrows():
                    f.write(f"\nRemoving Layer {int(row['removed_layer'])} "
                           f"({int(row['from_n_layers'])}-layer → {int(row['to_n_layers'])}-layer):\n")
                    f.write(f"  Frequency shift: {row['freq_shift_hz']:+.3f} Hz "
                           f"({row['original_peak_freq']:.2f} → {row['new_peak_freq']:.2f} Hz)\n")
                    f.write(f"  Amplitude change: {row['amp_change']:+.3f} "
                           f"({row['original_peak_amp']:.2f} → {row['new_peak_amp']:.2f})\n")
                    f.write(f"  Spectral energy change: {row['spectral_energy_change']:.4f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"✅ Analysis report saved: {output_path}")

    def export_data_csv(self, output_path: Path):
        """
        Export all stripping data to CSV for external analysis.

        Parameters:
            output_path: Path for output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_records = []
        for step_data in self.steps_data:
            data_records.append({
                'step': step_data['step'],
                'n_layers': step_data['n_layers'],
                'peak_frequency_hz': step_data['peak_freq'],
                'peak_amplitude': step_data['peak_amp'],
                'step_name': step_data['step_name']
            })

        df = pd.DataFrame(data_records)
        df.to_csv(output_path, index=False)
        print(f"✅ Data exported to CSV: {output_path}")


def analyze_strip_directory(strip_dir: Path, output_dir: Optional[Path] = None) -> Dict:
    """
    Convenience function to analyze a strip directory.

    Parameters:
        strip_dir: Path to strip directory
        output_dir: Optional output directory for reports

    Returns:
        Dictionary with analysis results
    """
    analyzer = StrippingAnalyzer(strip_dir)

    # Compute all analyses
    stats = analyzer.compute_statistics()
    controlling = analyzer.detect_controlling_interfaces()
    contributions = analyzer.analyze_layer_contributions()

    # Generate reports if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer.generate_analysis_report(output_dir / "advanced_analysis_report.txt")
        analyzer.export_data_csv(output_dir / "stripping_data.csv")

        # Save contributions dataframe
        if not contributions.empty:
            contributions.to_csv(output_dir / "layer_contributions.csv", index=False)

    return {
        'statistics': stats,
        'controlling_interfaces': controlling,
        'layer_contributions': contributions,
        'analyzer': analyzer
    }


__all__ = [
    "StrippingAnalyzer",
    "analyze_strip_directory"
]
