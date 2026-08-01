"""``LayerModel`` — the tool-aware view-model for the right-dock Layers rail.

The agreed signature move: a strip run's **STEPS are the layers**
(``step::<i>`` — "Step0 · 6-layer" …), and Forward-Multiple's **profiles**
are layers (``prof::<name>``) — keyed, toggleable, stylable curve items the
canvases render.  Sits beside :class:`AppState` (the data authority): it
holds per-key visibility + display settings and re-derives the tree from the
session's results per active tool.  GUI-only; reads results through AppState
accessors, never mutates the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from HV_Strip_Progressive.gui.v2.state.app_state import AppState
from HV_Strip_Progressive.gui.v2.state.tool import StripTool

KIND_GROUP = "group"
KIND_MODEL = "model"      # a per-step / per-profile curve (stylable)
KIND_PEAK = "peak"        # the peak-markers toggle

#: Sequential per-item hues (steps ramp through these; the user can override).
STEP_COLORS = (
    "#C87A20", "#2E86AB", "#3E7A45", "#9C27B0", "#C8102E",
    "#00838F", "#6D4C41", "#546E7A", "#AD1457", "#4527A0",
)

_DEFAULT_DISPLAY = {
    "opacity": 1.0,
    "color": "#C87A20",     # the HV Strip amber accent
    "line_width": 2,
    "line_style": "solid",
    "show_label": True,
}


@dataclass
class LayerNode:
    """One row in the layers tree."""

    key: str
    label: str
    kind: str
    children: List["LayerNode"] = field(default_factory=list)
    detail: str = ""       # column-2 text (e.g. f0)
    color: str = ""        # chip colour


class LayerModel(QObject):
    """Tool-aware per-layer visibility + display settings."""

    visibility_changed = Signal(str)
    display_changed = Signal(str)
    layers_rebuilt = Signal()

    def __init__(self, app_state: AppState, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._app_state = app_state
        self._tool: StripTool = app_state.active_tool
        self._groups: List[LayerNode] = []
        self._kind: Dict[str, str] = {}
        self._available: set = set()
        self._visible: Dict[str, bool] = {}
        self._display: Dict[str, Dict[str, Any]] = {}

        app_state.session_opened.connect(self._on_session_opened)
        for signal in (
            app_state.profiles_changed,
            app_state.forward_changed,
            app_state.strip_changed,
            app_state.research_changed,
        ):
            signal.connect(self.rebuild)
        app_state.active_tool_changed.connect(self.set_tool)
        self.rebuild()

    # ------------------------------------------------------------------
    def set_tool(self, tool: StripTool) -> None:
        if tool is self._tool:
            return
        self._tool = tool
        self.rebuild()

    @property
    def tool(self) -> StripTool:
        return self._tool

    # ------------------------------------------------------------------
    def _on_session_opened(self) -> None:
        self._visible.clear()
        self._display.clear()
        self.rebuild()

    def rebuild(self) -> None:
        self._groups = []
        self._kind = {}
        self._available = set()
        a = self._app_state.analysis
        if a is None:
            self.layers_rebuilt.emit()
            return

        if self._tool is StripTool.FORWARD:
            nodes: List[LayerNode] = []
            for i, (name, res) in enumerate(
                    self._app_state.forward_results().items()):
                key = f"prof::{name}"
                color = STEP_COLORS[i % len(STEP_COLORS)]
                self._display.setdefault(key, {}).setdefault("color", color)
                f0 = ""
                peaks = getattr(res, "peaks", None) or []
                if peaks:
                    f0 = f"{peaks[0].frequency:.2f} Hz"
                nodes.append(LayerNode(key, name, KIND_MODEL,
                                       detail=f0, color=color))
            if nodes:
                self._add(LayerNode("profiles", "Profiles", KIND_GROUP, nodes))
                self._add(LayerNode("peaks", "Peak markers", KIND_PEAK))

        elif self._tool is StripTool.STRIP:
            # The ACTIVE strip result's steps (first/only result for now;
            # the Strip panel selects which result is active in P5).
            strips = self._app_state.strip_results()
            for _name, result in list(strips.items())[:1]:
                nodes = []
                for i, step in enumerate(getattr(result, "steps", []) or []):
                    key = f"step::{step.step_number}"
                    color = STEP_COLORS[i % len(STEP_COLORS)]
                    self._display.setdefault(key, {}).setdefault("color", color)
                    nodes.append(LayerNode(
                        key,
                        f"Step{step.step_number} · {step.n_layers}-layer",
                        KIND_MODEL,
                        detail=(f"{step.peak_frequency:.2f} Hz"
                                if step.peak_frequency else ""),
                        color=color,
                    ))
                if nodes:
                    self._add(LayerNode("steps", "Strip steps", KIND_GROUP,
                                        nodes))
                    self._add(LayerNode("peaks", "Peak markers", KIND_PEAK))

        elif self._tool is StripTool.RESEARCH:
            # The study's generated figures — keyed, toggleable gallery items.
            nodes = []
            for i, (name, _path) in enumerate(
                    self._app_state.research_figures()):
                key = f"fig::{name}"
                color = STEP_COLORS[i % len(STEP_COLORS)]
                nodes.append(LayerNode(key, name, KIND_MODEL, color=color))
            if nodes:
                self._add(LayerNode("figures", "Study figures", KIND_GROUP,
                                    nodes))

        self.layers_rebuilt.emit()

    def _add(self, node: LayerNode) -> None:
        self._register(node)
        self._groups.append(node)

    def _register(self, node: LayerNode) -> None:
        self._available.add(node.key)
        self._kind[node.key] = node.kind
        self._visible.setdefault(node.key, True)
        for child in node.children:
            self._register(child)

    # ------------------------------------------------------------------
    #  Query
    # ------------------------------------------------------------------
    def groups(self) -> List[LayerNode]:
        return list(self._groups)

    def find_node(self, key: str) -> Optional[LayerNode]:
        def _walk(nodes: List[LayerNode]) -> Optional[LayerNode]:
            for n in nodes:
                if n.key == key:
                    return n
                hit = _walk(n.children)
                if hit is not None:
                    return hit
            return None

        return _walk(self._groups)

    def descendant_keys(self, key: str) -> List[str]:
        node = self.find_node(key)
        if node is None:
            return []
        out: List[str] = []

        def _collect(n: LayerNode) -> None:
            for c in n.children:
                out.append(c.key)
                _collect(c)

        _collect(node)
        return out

    def is_available(self, key: str) -> bool:
        return key in self._available

    def kind(self, key: str) -> str:
        return self._kind.get(key, KIND_GROUP)

    def is_visible(self, key: str) -> bool:
        return self._visible.get(key, True)

    def display(self, key: str) -> Dict[str, Any]:
        merged = dict(_DEFAULT_DISPLAY)
        merged.update(self._display.get(key, {}))
        return merged

    # ------------------------------------------------------------------
    #  Mutate
    # ------------------------------------------------------------------
    def set_visible(self, key: str, visible: bool) -> None:
        if self._visible.get(key, True) == visible:
            return
        self._visible[key] = visible
        self.visibility_changed.emit(key)

    def set_display(self, key: str, **kwargs: Any) -> None:
        current = self._display.setdefault(key, {})
        changed = False
        for name, value in kwargs.items():
            if current.get(name) != value:
                current[name] = value
                changed = True
        if changed:
            self.display_changed.emit(key)


__all__ = [
    "KIND_GROUP", "KIND_MODEL", "KIND_PEAK", "STEP_COLORS",
    "LayerModel", "LayerNode",
]
