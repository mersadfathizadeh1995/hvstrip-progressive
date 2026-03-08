"""
Strip worker — runs the full HV stripping workflow in a QThread.
"""

from PyQt5.QtCore import QThread, pyqtSignal


class StripWorker(QThread):
    """Runs the complete stripping workflow for a single profile."""

    finished = pyqtSignal(dict)   # strip results dict
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # (percent, message)
    step_done = pyqtSignal(int, dict) # (step_index, step_result) for live updates

    def __init__(self, profile, engine_name: str, engine_config: dict,
                 freq_config: dict, peak_config: dict,
                 dual_resonance_config: dict = None,
                 generate_report: bool = True,
                 output_dir: str = '',
                 parent=None):
        super().__init__(parent)
        self._profile = profile
        self._engine_name = engine_name
        self._engine_config = engine_config
        self._freq_config = freq_config
        self._peak_config = peak_config
        self._dual_config = dual_resonance_config or {}
        self._gen_report = generate_report
        self._output_dir = output_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import sys, os
            pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            parent = os.path.dirname(pkg)
            hvstrip_root = os.path.join(parent, 'hvstrip_progressive')
            if hvstrip_root not in sys.path:
                sys.path.insert(0, hvstrip_root)

            from core.engines import EngineRegistry
            from core.peak_detection import detect_peaks

            registry = EngineRegistry()
            engine = registry.get(self._engine_name)
            if engine is None:
                self.error.emit(f'Engine "{self._engine_name}" not found')
                return

            # Build stripping steps
            layers = list(self._profile.layers)
            n_steps = len([l for l in layers if not l.is_halfspace])
            results = {'steps': [], 'profile': self._profile, 'engine': self._engine_name}

            for step in range(n_steps + 1):
                if self._cancelled:
                    self.progress.emit(0, 'Cancelled')
                    return

                pct = int((step / max(n_steps, 1)) * 100)
                remaining = len(layers) - step
                self.progress.emit(pct, f'Step {step}/{n_steps} — {remaining} layers')

                # Build profile for this step
                step_layers = layers[step:]
                from core.soil_profile import SoilProfile
                step_profile = SoilProfile(step_layers, name=f'Step {step}')

                # Forward model
                config = {**self._engine_config, **self._freq_config}
                result = engine.compute_from_profile(step_profile, config)
                if result is None:
                    continue

                # Peak detection
                peak_info = detect_peaks(
                    result.frequencies, result.amplitudes,
                    **self._peak_config
                )

                step_result = {
                    'step': step,
                    'profile': step_profile,
                    'freqs': result.frequencies,
                    'amps': result.amplitudes,
                    'f0': peak_info.get('f0'),
                    'f0_amp': peak_info.get('f0_amplitude'),
                    'all_peaks': peak_info.get('all_peaks', []),
                    'removed_layer': layers[step - 1] if step > 0 else None,
                }
                results['steps'].append(step_result)
                self.step_done.emit(step, step_result)

            # Dual-resonance
            if self._dual_config.get('enabled') and len(results['steps']) >= 2:
                s0 = results['steps'][0]
                s1 = results['steps'][1]
                if s0.get('f0') and s1.get('f0'):
                    ratio = max(s0['f0'], s1['f0']) / max(min(s0['f0'], s1['f0']), 1e-6)
                    shift = abs(s0['f0'] - s1['f0'])
                    results['dual_resonance'] = {
                        'f0': s0['f0'], 'f1': s1['f0'],
                        'ratio': ratio, 'shift': shift,
                        'success': (ratio >= self._dual_config.get('separation_ratio', 2.0)
                                    and shift >= self._dual_config.get('min_shift', 0.2)),
                    }

            # Controlling interface
            if len(results['steps']) >= 2:
                shifts = []
                for i in range(1, len(results['steps'])):
                    prev = results['steps'][i - 1]
                    curr = results['steps'][i]
                    if prev.get('f0') and curr.get('f0'):
                        shifts.append({
                            'step': i, 'shift': abs(curr['f0'] - prev['f0']),
                            'removed': prev.get('removed_layer'),
                        })
                if shifts:
                    ctrl = max(shifts, key=lambda s: s['shift'])
                    results['controlling_interface'] = ctrl

            self.progress.emit(100, 'Complete')
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))
