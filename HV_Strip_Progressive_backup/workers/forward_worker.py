"""
Forward model worker — runs a single HV forward computation in a QThread.
"""

from PyQt5.QtCore import QThread, pyqtSignal


class ForwardWorker(QThread):
    """Computes a single HV curve from a SoilProfile."""

    finished = pyqtSignal(object)  # EngineResult or None
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, profile, engine_name: str, engine_config: dict,
                 freq_config: dict, parent=None):
        super().__init__(parent)
        self._profile = profile
        self._engine_name = engine_name
        self._engine_config = engine_config
        self._freq_config = freq_config

    def run(self):
        try:
            self.progress.emit(f'Computing {self._engine_name} forward model...')

            # Import core
            import sys, os
            pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            parent = os.path.dirname(pkg)
            hvstrip_root = os.path.join(parent, 'hvstrip_progressive')
            if hvstrip_root not in sys.path:
                sys.path.insert(0, hvstrip_root)

            from core.engines import EngineRegistry
            registry = EngineRegistry()
            engine = registry.get(self._engine_name)
            if engine is None:
                self.error.emit(f'Engine "{self._engine_name}" not found')
                return

            result = engine.compute_from_profile(
                self._profile,
                {**self._engine_config, **self._freq_config}
            )
            self.progress.emit('Forward model complete.')
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
