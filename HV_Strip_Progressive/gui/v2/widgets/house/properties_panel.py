"""``PropertiesPanel`` — per-layer display + info for the right rail.

Adapted (copy-not-import) from invert_hvsr's Results properties: stylable
nodes (per-STEP / per-PROFILE curves, ``KIND_MODEL``) get a colour picker +
line style/width/opacity + a read-only info block (f₀ · A₀ · Vs30 · layers,
straight off the api result objects via AppState accessors); group/peak
nodes get a context block (never a blank panel).  Edits write the
:class:`LayerModel` display dict (the canvases re-pen from
``display_changed``) — never the session.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from HV_Strip_Progressive.gui.v2.state.app_state import AppState
from HV_Strip_Progressive.gui.v2.state.layer_model import (
    KIND_GROUP,
    KIND_MODEL,
    KIND_PEAK,
    LayerModel,
)
from HV_Strip_Progressive.gui.v2.widgets.house.color_swatch import (
    ColorSwatchButton,
)

_LINE_STYLES = [("Solid", "solid"), ("Dashed", "dash"), ("Dotted", "dot")]


class PropertiesPanel(QWidget):
    """Display editor + info block bound to the layer view-model."""

    def __init__(
        self,
        layer_model: LayerModel,
        app_state: AppState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._model = layer_model
        self._app = app_state
        self._key = ""
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._title = QLabel("")
        self._title.setProperty("role", "heading")
        self._title.setWordWrap(True)
        outer.addWidget(self._title)

        self._empty = QLabel("Select a layer to edit its display settings.")
        self._empty.setProperty("role", "muted")
        self._empty.setWordWrap(True)
        outer.addWidget(self._empty)

        # ── Style form ──
        self._form = QWidget(self)
        form = QFormLayout(self._form)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self._color = ColorSwatchButton()
        self._color.color_changed.connect(lambda _c: self._on_changed())
        crow = QHBoxLayout()
        crow.setContentsMargins(0, 0, 0, 0)
        crow.addWidget(self._color)
        crow.addStretch(1)
        cwrap = QWidget()
        cwrap.setLayout(crow)
        form.addRow("Colour", cwrap)
        self._style = QComboBox()
        for label, _v in _LINE_STYLES:
            self._style.addItem(label)
        self._style.currentIndexChanged.connect(self._on_changed)
        form.addRow("Style", self._style)
        self._width = QSpinBox()
        self._width.setRange(1, 8)
        self._width.valueChanged.connect(self._on_changed)
        form.addRow("Width", self._width)
        self._opacity = QSlider(Qt.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.valueChanged.connect(self._on_changed)
        self._opacity_lbl = QLabel("100%")
        self._opacity_lbl.setMinimumWidth(38)
        orow = QHBoxLayout()
        orow.setContentsMargins(0, 0, 0, 0)
        orow.addWidget(self._opacity, 1)
        orow.addWidget(self._opacity_lbl)
        owrap = QWidget()
        owrap.setLayout(orow)
        form.addRow("Opacity", owrap)
        outer.addWidget(self._form)

        # ── Info block ──
        self._info_title = QLabel("Info")
        self._info_title.setProperty("role", "caption")
        outer.addWidget(self._info_title)
        self._info = QWidget(self)
        info_form = QFormLayout(self._info)
        info_form.setContentsMargins(0, 0, 0, 0)
        info_form.setSpacing(4)
        self._info_rows: Dict[str, QLabel] = {}
        for key, label in (("f0", "f₀ (Hz)"), ("a0", "A₀"),
                           ("vs30", "Vs30 (m/s)"), ("layers", "Layers")):
            val = QLabel("—")
            val.setProperty("role", "muted")
            info_form.addRow(label, val)
            self._info_rows[key] = val
        outer.addWidget(self._info)

        # ── Context block (groups / peaks) ──
        self._context = QLabel("")
        self._context.setProperty("role", "muted")
        self._context.setWordWrap(True)
        self._context.setVisible(False)
        outer.addWidget(self._context)
        self._grp_row = QWidget(self)
        grow = QHBoxLayout(self._grp_row)
        grow.setContentsMargins(0, 0, 0, 0)
        show_btn = QPushButton("Show all")
        show_btn.clicked.connect(lambda: self._set_group_visible(True))
        hide_btn = QPushButton("Hide all")
        hide_btn.clicked.connect(lambda: self._set_group_visible(False))
        grow.addWidget(show_btn)
        grow.addWidget(hide_btn)
        grow.addStretch(1)
        self._grp_row.setVisible(False)
        outer.addWidget(self._grp_row)

        outer.addStretch(1)
        self._model.layers_rebuilt.connect(self._on_rebuilt)
        self.set_layer("")

    # ------------------------------------------------------------------
    def set_layer(self, key: str) -> None:
        self._key = key or ""
        kind = self._model.kind(self._key) if self._key else ""
        styleable = kind == KIND_MODEL

        self._form.setVisible(styleable)
        self._info_title.setVisible(styleable)
        self._info.setVisible(styleable)
        self._context.setVisible(False)
        self._grp_row.setVisible(False)
        self._empty.setVisible(not self._key)

        if not self._key:
            self._title.setText("")
            return

        node = self._model.find_node(self._key)
        self._title.setText(node.label if node else self._key)

        if styleable:
            disp = self._model.display(self._key)
            self._loading = True
            self._color.set_color(disp.get("color") or "")
            for i, (_l, v) in enumerate(_LINE_STYLES):
                if v == disp.get("line_style", "solid"):
                    self._style.setCurrentIndex(i)
            self._width.setValue(int(disp.get("line_width", 2)))
            self._opacity.setValue(int(round(disp.get("opacity", 1.0) * 100)))
            self._opacity_lbl.setText(f"{self._opacity.value()}%")
            self._loading = False
            self._fill_info()
            return

        if kind == KIND_GROUP:
            n = len(self._model.descendant_keys(self._key))
            self._context.setText(f"{n} item(s)")
            self._context.setVisible(True)
            self._grp_row.setVisible(True)
        elif kind == KIND_PEAK:
            self._context.setText(
                "The checkbox toggles the peak markers on the canvas.")
            self._context.setVisible(True)

    # ------------------------------------------------------------------
    def _fill_info(self) -> None:
        for v in self._info_rows.values():
            v.setText("—")
        facts = self._facts_for(self._key)
        for name, value in facts.items():
            if name in self._info_rows and value is not None:
                self._info_rows[name].setText(
                    f"{value:.3f}" if isinstance(value, float) else str(value))

    def _facts_for(self, key: str) -> Dict[str, Any]:
        if key.startswith("step::"):
            try:
                idx = int(key.split("::")[1])
            except (ValueError, IndexError):
                return {}
            for result in self._app.strip_results().values():
                for step in getattr(result, "steps", []) or []:
                    if step.step_number == idx:
                        return {
                            "f0": float(step.peak_frequency or 0) or None,
                            "a0": float(step.peak_amplitude or 0) or None,
                            "layers": step.n_layers or None,
                        }
        if key.startswith("prof::"):
            name = key[len("prof::"):]
            res = self._app.forward_results().get(name)
            if res is not None:
                peaks = getattr(res, "peaks", None) or []
                out: Dict[str, Any] = {}
                if peaks:
                    out["f0"] = float(peaks[0].frequency)
                    out["a0"] = float(peaks[0].amplitude)
                return out
        return {}

    def _on_changed(self, *_a) -> None:
        if self._loading or not self._key:
            return
        self._opacity_lbl.setText(f"{self._opacity.value()}%")
        self._model.set_display(
            self._key,
            color=self._color.color(),
            line_style=_LINE_STYLES[self._style.currentIndex()][1],
            line_width=self._width.value(),
            opacity=self._opacity.value() / 100.0,
        )

    def _set_group_visible(self, visible: bool) -> None:
        for k in self._model.descendant_keys(self._key):
            self._model.set_visible(k, visible)
        if self._key:
            self._model.set_visible(self._key, visible)

    def _on_rebuilt(self) -> None:
        if self._key and not self._model.is_available(self._key):
            self.set_layer("")
        elif self._key:
            self.set_layer(self._key)


__all__ = ["PropertiesPanel"]
