"""
Batch worker — processes multiple profiles through the stripping workflow.
"""

import os
from PyQt5.QtCore import QThread, pyqtSignal


class BatchWorker(QThread):
    """Runs the HV strip workflow for multiple profile files."""

    finished = pyqtSignal(list)         # list of per-profile results
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)     # (percent, message)
    profile_done = pyqtSignal(int, dict) # (index, result_dict)

    def __init__(self, file_paths: list, engine_name: str,
                 engine_config: dict, freq_config: dict,
                 peak_config: dict, batch_config: dict = None,
                 output_dir: str = '', parent=None):
        super().__init__(parent)
        self._files = file_paths
        self._engine_name = engine_name
        self._engine_config = engine_config
        self._freq_config = freq_config
        self._peak_config = peak_config
        self._batch_config = batch_config or {}
        self._output_dir = output_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import sys
            pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            parent = os.path.dirname(pkg)
            hvstrip_root = os.path.join(parent, 'hvstrip_progressive')
            if hvstrip_root not in sys.path:
                sys.path.insert(0, hvstrip_root)

            from core.soil_profile import SoilProfile
            from core.engines import EngineRegistry
            from core.peak_detection import detect_peaks

            registry = EngineRegistry()
            engine = registry.get(self._engine_name)
            if engine is None:
                self.error.emit(f'Engine "{self._engine_name}" not found')
                return

            n = len(self._files)
            all_results = []

            for i, fpath in enumerate(self._files):
                if self._cancelled:
                    self.progress.emit(0, 'Cancelled')
                    return

                name = os.path.splitext(os.path.basename(fpath))[0]
                pct = int((i / max(n, 1)) * 100)
                self.progress.emit(pct, f'[{i+1}/{n}] {name}')

                try:
                    profile = SoilProfile.from_auto(fpath)
                    valid, errors = profile.validate()
                    if not valid:
                        all_results.append({
                            'name': name, 'path': fpath,
                            'success': False, 'error': '; '.join(errors),
                        })
                        continue

                    # Run stripping
                    layers = list(profile.layers)
                    n_steps = len([l for l in layers if not l.is_halfspace])
                    steps = []

                    for step in range(n_steps + 1):
                        if self._cancelled:
                            return
                        step_layers = layers[step:]
                        step_prof = SoilProfile(step_layers, name=f'{name}_Step{step}')
                        config = {**self._engine_config, **self._freq_config}
                        result = engine.compute_from_profile(step_prof, config)
                        if result is None:
                            continue
                        peak_info = detect_peaks(
                            result.frequencies, result.amplitudes,
                            **self._peak_config
                        )
                        steps.append({
                            'step': step,
                            'freqs': result.frequencies,
                            'amps': result.amplitudes,
                            'f0': peak_info.get('f0'),
                            'f0_amp': peak_info.get('f0_amplitude'),
                        })

                    profile_result = {
                        'name': name, 'path': fpath,
                        'success': True, 'steps': steps,
                        'profile': profile,
                    }

                    # Save per-profile output
                    if self._output_dir:
                        self._save_profile_output(name, profile_result)

                    all_results.append(profile_result)
                    self.profile_done.emit(i, profile_result)

                except Exception as e:
                    all_results.append({
                        'name': name, 'path': fpath,
                        'success': False, 'error': str(e),
                    })

            self.progress.emit(100, f'Batch complete — {len(all_results)} profiles')
            self.finished.emit(all_results)

        except Exception as e:
            self.error.emit(str(e))

    def _save_profile_output(self, name: str, result: dict):
        import csv
        d = os.path.join(self._output_dir, name)
        os.makedirs(d, exist_ok=True)

        for step_data in result.get('steps', []):
            step = step_data['step']
            csv_path = os.path.join(d, f'step_{step}_hv_curve.csv')
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frequency_hz', 'amplitude'])
                if step_data['freqs'] is not None:
                    for fr, am in zip(step_data['freqs'], step_data['amps']):
                        w.writerow([f'{fr:.6f}', f'{am:.6f}'])

        # Summary
        with open(os.path.join(d, 'summary.txt'), 'w') as f:
            f.write(f'Profile: {name}\n')
            f.write(f'Steps: {len(result.get("steps", []))}\n')
            for s in result.get('steps', []):
                f0 = s.get('f0', 'N/A')
                f0_str = f'{f0:.4f} Hz' if isinstance(f0, (int, float)) else str(f0)
                f.write(f'  Step {s["step"]}: f0 = {f0_str}\n')
