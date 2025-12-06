"""
Workflow Worker Thread
Handles background execution of HVSTRIP workflows
"""

import traceback
from pathlib import Path
from PySide6.QtCore import QThread, Signal

from hvstrip_progressive.core import batch_workflow, stripper, hv_forward, hv_postprocess
from hvstrip_progressive.core.report_generator import ProgressiveStrippingReporter


class WorkflowWorker(QThread):
    """Worker thread for running HVSTRIP workflows in background"""

    # Signals
    progress_updated = Signal(int, str)  # value, message
    log_message = Signal(str)
    workflow_completed = Signal(str)  # output_dir
    workflow_failed = Signal(str)  # error_message

    def __init__(self, model_file, exe_path, output_dir, mode, config):
        super().__init__()

        self.model_file = model_file
        self.exe_path = exe_path
        self.output_dir = output_dir
        self.mode = mode
        self.config = config
        self.should_stop = False

    def run(self):
        """Execute the workflow"""
        try:
            if self.mode == "Complete Workflow":
                self.run_complete_workflow()
            elif self.mode == "Strip Only":
                self.run_strip_only()
            elif self.mode == "Forward Only":
                self.run_forward_only()
            elif self.mode == "Postprocess Only":
                self.run_postprocess_only()

            if not self.should_stop:
                self.workflow_completed.emit(self.output_dir)

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.workflow_failed.emit(error_msg)

    def stop(self):
        """Request worker to stop"""
        self.should_stop = True

    def run_complete_workflow(self):
        """Run complete workflow: strip, forward, postprocess, report"""
        self.log_message.emit("Starting complete workflow...")
        self.progress_updated.emit(0, "Initializing...")

        # Build workflow configuration
        workflow_config = self.build_workflow_config()

        # Run workflow using batch_workflow module
        self.log_message.emit(f"Model file: {self.model_file}")
        self.log_message.emit(f"Output directory: {self.output_dir}")

        # Step 1: Layer stripping
        if self.should_stop:
            return

        self.progress_updated.emit(10, "Performing layer stripping...")
        self.log_message.emit("\n=== Layer Stripping ===")

        strip_dir = Path(self.output_dir) / "strip"
        model_info = stripper.strip_layers(self.model_file, str(strip_dir))

        self.log_message.emit(f"Created {len(model_info['step_dirs'])} peeled models")

        # Step 2: Forward modeling
        if self.should_stop:
            return

        self.progress_updated.emit(30, "Running forward modeling...")
        self.log_message.emit("\n=== Forward Modeling ===")

        for i, step_dir in enumerate(model_info['step_dirs']):
            if self.should_stop:
                return

            step_num = i
            progress = 30 + int((i / len(model_info['step_dirs'])) * 40)
            self.progress_updated.emit(progress, f"Computing HV for Step {step_num}...")

            model_file = Path(step_dir) / "model.txt"
            hv_file = Path(step_dir) / "hv_curve.csv"

            self.log_message.emit(f"Step {step_num}: Computing HV curve...")

            # Run forward modeling
            hv_forward.compute_hv_curve(
                model_file=str(model_file),
                output_file=str(hv_file),
                exe_path=self.exe_path,
                fmin=workflow_config['fmin'],
                fmax=workflow_config['fmax'],
                nf=workflow_config['nf'],
                nmr=workflow_config.get('nmr', 10),
                nml=workflow_config.get('nml', 10),
                nks=workflow_config.get('nks', 10),
                adaptive_config=workflow_config.get('adaptive_scanning', {})
            )

            self.log_message.emit(f"Step {step_num}: HV curve saved to {hv_file}")

        # Step 3: Post-processing
        if self.should_stop:
            return

        self.progress_updated.emit(70, "Generating visualizations...")
        self.log_message.emit("\n=== Post-Processing ===")

        postprocess_config = self.build_postprocess_config()

        for i, step_dir in enumerate(model_info['step_dirs']):
            if self.should_stop:
                return

            step_num = i
            progress = 70 + int((i / len(model_info['step_dirs'])) * 15)
            self.progress_updated.emit(progress, f"Creating plots for Step {step_num}...")

            model_file = Path(step_dir) / "model.txt"
            hv_file = Path(step_dir) / "hv_curve.csv"

            self.log_message.emit(f"Step {step_num}: Generating plots...")

            # Run post-processing
            hv_postprocess.postprocess_step(
                hv_csv_file=str(hv_file),
                model_file=str(model_file),
                output_dir=str(step_dir),
                config=postprocess_config
            )

            self.log_message.emit(f"Step {step_num}: Plots saved")

        # Step 4: Report generation
        if self.should_stop:
            return

        self.progress_updated.emit(85, "Generating comprehensive reports...")
        self.log_message.emit("\n=== Report Generation ===")

        report_config = self.config.get('reports', {})
        if report_config.get('generate_reports') or report_config.get('generate_visualizations'):
            reports_dir = Path(self.output_dir) / "reports"
            reports_dir.mkdir(exist_ok=True)

            self.log_message.emit("Creating comprehensive reports...")

            reporter = ProgressiveStrippingReporter(
                strip_directory=str(strip_dir),
                output_dir=str(reports_dir)
            )

            report_files = reporter.generate_comprehensive_report(
                formats=[report_config.get('publication_format', 'png')]
            )

            self.log_message.emit(f"Generated {len(report_files)} report files")
            for file in report_files:
                self.log_message.emit(f"  - {Path(file).name}")

        self.progress_updated.emit(100, "Complete!")
        self.log_message.emit("\n=== Workflow Complete ===")

    def run_strip_only(self):
        """Run only layer stripping"""
        self.log_message.emit("Running layer stripping only...")
        self.progress_updated.emit(10, "Stripping layers...")

        strip_dir = Path(self.output_dir) / "strip"
        model_info = stripper.strip_layers(self.model_file, str(strip_dir))

        self.log_message.emit(f"Created {len(model_info['step_dirs'])} peeled models")
        self.progress_updated.emit(100, "Complete!")

    def run_forward_only(self):
        """Run only forward modeling (requires existing model file)"""
        self.log_message.emit("Running forward modeling only...")
        self.progress_updated.emit(10, "Computing HV curve...")

        workflow_config = self.build_workflow_config()

        hv_file = Path(self.output_dir) / "hv_curve.csv"

        hv_forward.compute_hv_curve(
            model_file=self.model_file,
            output_file=str(hv_file),
            exe_path=self.exe_path,
            fmin=workflow_config['fmin'],
            fmax=workflow_config['fmax'],
            nf=workflow_config['nf'],
            nmr=workflow_config.get('nmr', 10),
            nml=workflow_config.get('nml', 10),
            nks=workflow_config.get('nks', 10),
            adaptive_config=workflow_config.get('adaptive_scanning', {})
        )

        self.log_message.emit(f"HV curve saved to {hv_file}")
        self.progress_updated.emit(100, "Complete!")

    def run_postprocess_only(self):
        """Run only post-processing (requires existing HV curve)"""
        self.log_message.emit("Running post-processing only...")
        self.progress_updated.emit(10, "Generating visualizations...")

        postprocess_config = self.build_postprocess_config()

        # Look for HV curve file
        hv_file = Path(self.output_dir) / "hv_curve.csv"
        if not hv_file.exists():
            raise FileNotFoundError(f"HV curve file not found: {hv_file}")

        hv_postprocess.postprocess_step(
            hv_csv_file=str(hv_file),
            model_file=self.model_file,
            output_dir=self.output_dir,
            config=postprocess_config
        )

        self.log_message.emit("Plots generated successfully")
        self.progress_updated.emit(100, "Complete!")

    def build_workflow_config(self):
        """Build workflow configuration from GUI settings"""
        config = {}

        # Frequency settings
        freq_settings = self.config.get('frequency', {})
        config['fmin'] = freq_settings.get('fmin', 0.2)
        config['fmax'] = freq_settings.get('fmax', 20.0)
        config['nf'] = freq_settings.get('nf', 71)
        config['nmr'] = freq_settings.get('nmr', 10)
        config['nml'] = freq_settings.get('nml', 10)
        config['nks'] = freq_settings.get('nks', 10)

        # Adaptive scanning
        config['adaptive_scanning'] = freq_settings.get('adaptive_scanning', {})

        return config

    def build_postprocess_config(self):
        """Build post-processing configuration from GUI settings"""
        config = {}

        # Peak detection settings
        peak_settings = self.config.get('peak_detection', {})
        config['peak_detection'] = peak_settings

        # Visualization settings
        viz_settings = self.config.get('visualization', {})
        config.update(viz_settings)

        return config
