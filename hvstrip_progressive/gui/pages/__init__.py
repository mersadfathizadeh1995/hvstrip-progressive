"""
Pages module for HVSR Progressive Layer Stripping GUI.

Reorganized structure:
- HomePage: Consolidated workflow and batch processing
- ProfileEditorPage: Soil profile creation and editing
- ForwardModelingPage: Enhanced HV forward modeling
- VisualizationPage: Figure generation and export
- SettingsPage: Application settings
"""

from .home_page import HomePage
from .profile_editor_page import ProfileEditorPage
from .forward_modeling_page import ForwardModelingPage
from .visualization_page import VisualizationPage
from .settings_page import SettingsPage

from .workflow_page import WorkflowPage
from .strip_page import StripPage
from .forward_page import ForwardPage
from .postprocess_page import PostprocessPage
from .report_page import ReportPage
from .batch_page import BatchPage
from .analysis_page import AnalysisPage

__all__ = [
    'HomePage',
    'ProfileEditorPage',
    'ForwardModelingPage',
    'VisualizationPage',
    'SettingsPage',
    'WorkflowPage',
    'StripPage',
    'ForwardPage',
    'PostprocessPage',
    'ReportPage',
    'BatchPage',
    'AnalysisPage',
]
