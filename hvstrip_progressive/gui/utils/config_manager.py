"""
Configuration Manager
Handles saving and loading of GUI configurations
"""

import json
from pathlib import Path
from PySide6.QtWidgets import QFileDialog


class ConfigManager:
    """Manager for saving and loading GUI configurations"""

    def __init__(self):
        self.default_dir = str(Path.home() / "Documents" / "HVSTRIP-Configs")
        Path(self.default_dir).mkdir(parents=True, exist_ok=True)

    def save_config_dialog(self, config, parent=None):
        """Show save dialog and save configuration"""
        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            "Save Configuration",
            str(Path(self.default_dir) / "hvstrip_config.json"),
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            return self.save_config(config, file_path)

        return False

    def load_config_dialog(self, parent=None):
        """Show open dialog and load configuration"""
        file_path, _ = QFileDialog.getOpenFileName(
            parent,
            "Load Configuration",
            self.default_dir,
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            return self.load_config(file_path)

        return None

    def save_config(self, config, file_path):
        """Save configuration to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

    def load_config(self, file_path):
        """Load configuration from file"""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return None

    def get_default_config(self):
        """Get default configuration"""
        return {
            'inputs': {
                'model_file': '',
                'exe_path': '',
                'output_dir': ''
            },
            'mode': 'Complete Workflow',
            'settings': {
                'frequency': {
                    'fmin': 0.2,
                    'fmax': 20.0,
                    'nf': 71,
                    'nmr': 10,
                    'nml': 10,
                    'nks': 10,
                    'adaptive_scanning': {
                        'enable': True,
                        'max_passes': 2,
                        'edge_margin_frac': 0.05,
                        'fmax_expand_factor': 2.0,
                        'fmin_shrink_factor': 0.5,
                        'fmax_limit': 60.0,
                        'fmin_limit': 0.05
                    }
                },
                'peak_detection': {
                    'method': 'find_peaks',
                    'selection': 'leftmost',
                    'prominence': 0.2,
                    'distance': 3,
                    'freq_min': 0.5,
                    'freq_max': None,
                    'min_rel_height': 0.25,
                    'exclude_first_n': 1
                },
                'visualization': {
                    'hv_curve': {
                        'x_axis_scale': 'log',
                        'y_axis_scale': 'log',
                        'y_compression': 1.5,
                        'smoothing': {
                            'enable': True,
                            'window_length': 9,
                            'poly_order': 3
                        },
                        'show_bands': True,
                        'freq_window_left': 0.3,
                        'freq_window_right': 3.0,
                        'figure_width': 12,
                        'figure_height': 6,
                        'dpi': 200
                    },
                    'vs_profile': {
                        'show': True,
                        'annotate_deepest': True,
                        'annotate_max_vs': True,
                        'annotate_f0': True
                    },
                    'output': {
                        'save_separate': True,
                        'save_combined': True
                    }
                },
                'reports': {
                    'generate_reports': {
                        'summary_csv': True,
                        'metadata_json': True,
                        'text_report': True
                    },
                    'generate_visualizations': {
                        'hv_overlay': True,
                        'peak_evolution': True,
                        'interface_analysis': True,
                        'waterfall': True,
                        'publication_figure': True
                    },
                    'publication_format': 'png'
                }
            }
        }
