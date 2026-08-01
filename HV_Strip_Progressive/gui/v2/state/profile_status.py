"""Per-profile processing status — the ProfilesPanel badge vocabulary.

Copied (copy-not-import) from HV Pro gui_v2's ``state/pipeline_stage.py``
``ProcessingStatus`` so the badge semantics + colours match the family app.
These are SEMANTIC status colours shared across the family, not theme accents.
"""

from __future__ import annotations

from enum import Enum


class ProcessingStatus(Enum):
    """Per-profile, per-tool run status (the ProfilesPanel badges)."""

    NOT_STARTED = "not_started"  # never run
    QUEUED = "queued"            # waiting in the runner queue
    RUNNING = "running"          # currently being processed
    DONE = "done"                # completed successfully
    FAILED = "failed"            # last run raised

    @property
    def color(self) -> str:
        return {
            ProcessingStatus.NOT_STARTED: "#7e8794",
            ProcessingStatus.QUEUED: "#f0a020",
            ProcessingStatus.RUNNING: "#2a82da",
            ProcessingStatus.DONE: "#27ae60",
            ProcessingStatus.FAILED: "#c0392b",
        }[self]


__all__ = ["ProcessingStatus"]
