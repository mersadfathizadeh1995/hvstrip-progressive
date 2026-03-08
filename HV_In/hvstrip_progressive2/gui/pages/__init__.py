"""
Pages module for HVSR Progressive Layer Stripping GUI.
"""

from .workflow_page import WorkflowPage
from .strip_page import StripPage
from .forward_page import ForwardPage
from .postprocess_page import PostprocessPage
from .report_page import ReportPage
from .batch_page import BatchPage
from .analysis_page import AnalysisPage
from .settings_page import SettingsPage
from .research_page import ResearchPage
from .parallel_batch_page import ParallelBatchPage

__all__ = [
    'WorkflowPage',
    'StripPage',
    'ForwardPage',
    'PostprocessPage',
    'ReportPage',
    'BatchPage',
    'AnalysisPage',
    'SettingsPage',
    'ResearchPage',
    'ParallelBatchPage'
]
