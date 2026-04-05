"""ForwardWorker — QThread for single HV forward model computation."""
from PyQt5.QtCore import QThread, pyqtSignal


class ForwardWorker(QThread):
    """Compute HV curve in a background thread.

    Emits
    -----
    finished_signal : tuple(list, list)
        (frequencies, amplitudes) on success.
    error : str
        Error message on failure.
    """

    finished_signal = pyqtSignal(tuple)
    error = pyqtSignal(str)

    def __init__(self, model_path, config=None, engine_name="diffuse_field", parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._config = config or {}
        self._engine_name = engine_name

    def run(self):
        try:
            from ...core.hv_forward import compute_hv_curve
            freqs, amps = compute_hv_curve(
                self._model_path,
                config=self._config,
                engine_name=self._engine_name,
            )
            self.finished_signal.emit((freqs, amps))
        except Exception as exc:
            self.error.emit(str(exc))
