"""ResearchWorker — QThread for running comparison study pipeline phases."""

from PyQt5.QtCore import QThread, pyqtSignal


class ResearchWorker(QThread):
    """Run comparison study phases in a background thread.

    Emits
    -----
    progress : (int, int, str)
        (current, total, message) for progress tracking.
    phase_complete : (str, dict)
        (phase_name, result_dict) after each pipeline phase.
    finished_signal : dict
        Full results on success.
    error : str
        Error message on failure.
    """

    progress = pyqtSignal(int, int, str)
    phase_complete = pyqtSignal(str, dict)
    finished_signal = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, runner, phase="full", parent=None):
        super().__init__(parent)
        self._runner = runner
        self._phase = phase
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            # Wire progress callback
            self._runner.set_progress_callback(self._on_progress)
            results = {}

            if self._phase == "full":
                results = self._run_full()
            elif self._phase == "profiles":
                results = self._runner.generate_profiles()
                self.phase_complete.emit("profiles", results)
            elif self._phase == "comparison":
                results = self._runner.run_comparison()
                self.phase_complete.emit("comparison", results)
            elif self._phase == "metrics":
                results = self._runner.compute_metrics()
                self.phase_complete.emit("metrics", results)
            elif self._phase == "field_validation":
                results = self._runner.run_field_validation()
                self.phase_complete.emit("field_validation", results)
            elif self._phase == "report":
                results = self._runner.generate_report()
                self.phase_complete.emit("report", results)
            else:
                self.error.emit(f"Unknown phase: {self._phase}")
                return

            if not self._cancelled:
                self.finished_signal.emit(results)

        except Exception as exc:
            self.error.emit(str(exc))

    def _run_full(self):
        """Run all phases sequentially with inter-phase signals."""
        all_results = {}

        # Phase 1: Profiles
        if self._cancelled:
            return all_results
        self.progress.emit(0, 5, "Generating profiles...")
        r = self._runner.generate_profiles()
        all_results["profiles"] = r
        self.phase_complete.emit("profiles", r)

        # Phase 2: Comparison
        if self._cancelled:
            return all_results
        self.progress.emit(1, 5, "Running forward comparison...")
        r = self._runner.run_comparison()
        all_results["comparison"] = r
        self.phase_complete.emit("comparison", r)

        # Phase 3: Metrics
        if self._cancelled:
            return all_results
        self.progress.emit(2, 5, "Computing metrics...")
        r = self._runner.compute_metrics()
        all_results["metrics"] = r
        self.phase_complete.emit("metrics", r)

        # Phase 4: Field validation (optional)
        if self._cancelled:
            return all_results
        self.progress.emit(3, 5, "Running field validation...")
        r = self._runner.run_field_validation()
        all_results["field_validation"] = r
        self.phase_complete.emit("field_validation", r)

        # Phase 5: Report
        if self._cancelled:
            return all_results
        self.progress.emit(4, 5, "Generating report...")
        r = self._runner.generate_report()
        all_results["report"] = r
        self.phase_complete.emit("report", r)

        self.progress.emit(5, 5, "Complete")
        return all_results

    def _on_progress(self, current, total, message):
        """Forward progress from runner to Qt signal."""
        self.progress.emit(current, total, message)
