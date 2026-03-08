"""BatchWorker — QThread for processing multiple profiles."""
import os
from PyQt5.QtCore import QThread, pyqtSignal


class BatchWorker(QThread):
    """Process multiple profiles sequentially.

    Emits progress(message, current, total) for each profile.
    """

    progress = pyqtSignal(str, int, int)
    finished_signal = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, file_paths, output_dir, config=None, parent=None):
        super().__init__(parent)
        self._file_paths = file_paths
        self._output_dir = output_dir
        self._config = config or {}
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        results = []
        total = len(self._file_paths)
        try:
            from core.batch_workflow import run_complete_workflow

            for i, fpath in enumerate(self._file_paths):
                if self._cancelled:
                    self.progress.emit("Cancelled by user.", i, total)
                    break

                name = os.path.splitext(os.path.basename(fpath))[0]
                profile_out = os.path.join(self._output_dir, name)
                os.makedirs(profile_out, exist_ok=True)

                self.progress.emit(f"Processing {name} ({i+1}/{total})...", i, total)
                try:
                    result = run_complete_workflow(fpath, profile_out, config=self._config)
                    results.append({"name": name, "path": fpath, "status": "success", "result": result})
                except Exception as e:
                    results.append({"name": name, "path": fpath, "status": "error", "error": str(e)})
                    self.progress.emit(f"Error on {name}: {e}", i, total)

            self.finished_signal.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))
