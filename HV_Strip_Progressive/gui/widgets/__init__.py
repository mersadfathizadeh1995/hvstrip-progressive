"""Widgets sub-package."""
from .plot_widget import MatplotlibWidget
from .profile_preview_widget import ProfilePreviewWidget
from .layer_table_widget import LayerTableWidget
from .collapsible_group import CollapsibleGroupBox

__all__ = [
    "MatplotlibWidget", "ProfilePreviewWidget", "LayerTableWidget",
    "CollapsibleGroupBox",
]
