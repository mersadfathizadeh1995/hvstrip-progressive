"""The three HV Strip TOOLS + per-tool status (the tool-switcher vocabulary).

The house workbench here is a **TOOL-COLLECTION** (archetype C, the agreed
design): Forward Model | HV Strip | Research are peers — no forced order, no
``[step]`` pipeline chain.  The switcher shows each tool's
:class:`ToolStatus`, **pulled** from :meth:`AppState.status_for(tool)` — the
switcher never decides state itself.  Linear mini-flows (the strip wizard,
the research phases) nest INSIDE tools as sub-stage containers.
"""

from __future__ import annotations

from enum import Enum
from typing import List


class StripTool(Enum):
    """The peer tools of the HV Strip workbench (a tool-collection).

    ``DATA`` is the Round-2 first stage: unified profile loading + the
    editable layer table — every other tool consumes what it loads.
    """

    DATA = "data"
    FORWARD = "forward"
    STRIP = "strip"
    RESEARCH = "research"

    @property
    def label(self) -> str:
        return {
            StripTool.DATA: "Data Input",
            StripTool.FORWARD: "Forward Model",
            StripTool.STRIP: "HV Strip",
            StripTool.RESEARCH: "Research",
        }[self]

    @property
    def subtitle(self) -> str:
        return {
            StripTool.DATA: "load & prepare profiles",
            StripTool.FORWARD: "forward H/V curves",
            StripTool.STRIP: "peel layers · track peaks",
            StripTool.RESEARCH: "comparison studies",
        }[self]


class ToolStatus(Enum):
    """Per-tool status driving the switcher pill accent (NOT a chain)."""

    IDLE = "idle"         # ready to use
    RUNNING = "running"   # an op for this tool is executing
    DONE = "done"         # this tool has results in the session
    ERROR = "error"       # the last op for this tool failed


TOOL_ORDER: List[StripTool] = list(StripTool)


__all__ = ["StripTool", "ToolStatus", "TOOL_ORDER"]
