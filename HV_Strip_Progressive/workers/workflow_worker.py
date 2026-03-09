"""WorkflowWorker — QThread for the complete stripping workflow."""
from PyQt5.QtCore import QThread, pyqtSignal


class WorkflowWorker(QThread):
    """Run the complete stripping workflow in a background thread.

    Calls core.batch_workflow.run_complete_workflow().
    """

    progress = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_path, output_dir, config=None, parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._output_dir = output_dir
        self._config = config or {}

    def run(self):
        try:
            from hvstrip_progressive.core.batch_workflow import run_complete_workflow

            self.progress.emit("Starting stripping workflow...")
            result = run_complete_workflow(
                self._model_path,
                self._output_dir,
                workflow_config=self._config,
            )
            self.progress.emit("Workflow completed.")
            self.finished_signal.emit(result if isinstance(result, dict) else {"result": result})
        except Exception as exc:
            self.error.emit(str(exc))
