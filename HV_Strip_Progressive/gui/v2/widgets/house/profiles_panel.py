"""``ProfilesPanel`` — the global left files rail (HV Pro gui_v2 pattern).

Adapted (copy-not-import) from HV Pro's ``panels/process/stations_panel.py``:
a checkable profile tree whose CHECKED rows are the RUN SET, with per-row
status badges painted by a delegate, a tri-state Select-all header, and an
Assign-settings-to-checked button.

The badge columns are **stage-aware** (the user's design): in a tool stage
(Forward / Strip / Research) the row shows that tool's ``[Set][Run]`` pair;
in the Data stage it shows the ``[Fwd][Str][Res]`` run-status overview —
each profile may be only forward-modeled, only stripped, only researched.

Present in EVERY stage (mounted leftmost in the shell, collapsible).
Talks ONLY to AppState.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from HV_Strip_Progressive.gui.v2.state import (
    AppState,
    ProcessingStatus,
    StripTool,
)

_SQUARE_PX = 12
_STATUS_COL_WIDTH = 34
_COLOR_SETT_DONE = "#27ae60"
_COLOR_GRAY = "#b9c0c7"


def _status_label(status: ProcessingStatus) -> str:
    return {
        ProcessingStatus.NOT_STARTED: "Not started",
        ProcessingStatus.QUEUED: "Queued",
        ProcessingStatus.RUNNING: "Running",
        ProcessingStatus.DONE: "Done",
        ProcessingStatus.FAILED: "Failed",
    }.get(status, str(status))


class _BadgeDelegate(QStyledItemDelegate):
    """Paint one small coloured square per status/sett cell."""

    def __init__(self, kind: str, parent=None) -> None:
        super().__init__(parent)
        assert kind in ("sett", "status")
        self._kind = kind

    def _color_for(self, payload) -> str:
        if self._kind == "sett":
            return _COLOR_SETT_DONE if bool(payload) else _COLOR_GRAY
        if isinstance(payload, ProcessingStatus) and \
                payload is not ProcessingStatus.NOT_STARTED:
            return payload.color
        return _COLOR_GRAY

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(_STATUS_COL_WIDTH, max(_SQUARE_PX + 6, 18))

    def paint(self, painter, option: QStyleOptionViewItem, index) -> None:  # noqa: N802
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)

        payload = index.data(Qt.UserRole)
        color = QColor(self._color_for(payload))
        rect: QRect = option.rect
        side = _SQUARE_PX
        sq = QRect(rect.x() + (rect.width() - side) // 2,
                   rect.y() + (rect.height() - side) // 2, side, side)

        painter.save()
        painter.setRenderHint(painter.RenderHint.Antialiasing, False)
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor("#5a5a5a"), 1))
        painter.drawRect(sq)
        if isinstance(payload, ProcessingStatus) and \
                payload is ProcessingStatus.FAILED:
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(sq.topLeft(), sq.bottomRight())
            painter.drawLine(sq.topRight(), sq.bottomLeft())
        painter.restore()


class ProfilesPanel(QWidget):
    """The checkable, badge-annotated profile rail (see module docstring)."""

    load_requested = Signal()            # ＋ Load… → jump to the Data stage
    assign_settings_requested = Signal(list)   # profile names (targets)

    def __init__(self, app_state: AppState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._app = app_state
        self._suppress = False
        #: (key, header, tool) per badge column, col index = pos + 1
        self._badge_cols: List[tuple] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # Toolbar row
        bar = QHBoxLayout()
        load_btn = QPushButton("＋ Load…")
        load_btn.setProperty("primary", "true")
        load_btn.clicked.connect(self.load_requested)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove)
        bar.addWidget(load_btn)
        bar.addWidget(remove_btn)
        bar.addStretch(1)
        outer.addLayout(bar)

        # Select-all header
        head = QFrame()
        head.setProperty("role", "subheader")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(2, 2, 2, 2)
        hl.setSpacing(6)
        self._select_all = QCheckBox("Select all")
        self._select_all.setTristate(True)
        self._select_all.clicked.connect(self._on_select_all)
        hl.addWidget(self._select_all)
        hl.addStretch(1)
        self._count_lbl = QLabel("")
        self._count_lbl.setProperty("role", "muted")
        hl.addWidget(self._count_lbl)
        outer.addWidget(head)

        # The tree
        self._tree = QTreeWidget()
        self._tree.setObjectName("layerTree")
        self._tree.setRootIsDecorated(False)
        self._tree.setUniformRowHeights(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_menu)
        outer.addWidget(self._tree, 1)

        # Assign settings
        self._assign_btn = QPushButton("Assign settings")
        self._assign_btn.clicked.connect(self._on_assign)
        outer.addWidget(self._assign_btn)

        # AppState wiring
        app_state.session_opened.connect(self.rebuild)
        app_state.profiles_changed.connect(self.rebuild)
        app_state.checked_changed.connect(lambda _n: self._sync_checks())
        app_state.focus_changed.connect(self._on_focus_changed)
        app_state.profile_status_changed.connect(self._on_status_changed)
        app_state.profile_settings_changed.connect(
            lambda name: self._refresh_row(name))
        app_state.active_tool_changed.connect(lambda _t: self._reconfigure())

        self._reconfigure()

    # ==================================================================
    #  Column configuration (stage-aware)
    # ==================================================================
    def _column_spec(self) -> List[tuple]:
        tool = self._app.active_tool
        if tool is StripTool.FORWARD:
            return [("sett", "Set", StripTool.FORWARD),
                    ("status", "Run", StripTool.FORWARD)]
        if tool is StripTool.STRIP:
            return [("sett", "Set", StripTool.STRIP),
                    ("status", "Run", StripTool.STRIP)]
        if tool is StripTool.RESEARCH:
            return [("sett", "Set", StripTool.RESEARCH),
                    ("status", "Run", StripTool.RESEARCH)]
        # Data stage (or anything else): the per-tool run overview.
        return [("status", "Fwd", StripTool.FORWARD),
                ("status", "Str", StripTool.STRIP),
                ("status", "Res", StripTool.RESEARCH)]

    def _reconfigure(self) -> None:
        """Rebuild columns for the active stage, then repopulate rows."""
        self._badge_cols = self._column_spec()
        headers = ["Profile"] + [h for _k, h, _t in self._badge_cols]
        self._tree.setColumnCount(len(headers))
        self._tree.setHeaderLabels(headers)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i, (kind, _h, _t) in enumerate(self._badge_cols, start=1):
            header.setSectionResizeMode(i, QHeaderView.Fixed)
            header.resizeSection(i, _STATUS_COL_WIDTH)
            self._tree.setItemDelegateForColumn(
                i, _BadgeDelegate(kind, self._tree))
        self.rebuild()

    # ==================================================================
    #  Rows
    # ==================================================================
    def rebuild(self) -> None:
        self._suppress = True
        try:
            self._tree.clear()
            checked = set(self._app.checked_profiles())
            for row in self._app.profiles() or []:
                name = row["name"]
                item = QTreeWidgetItem([name])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(
                    0, Qt.Checked if name in checked else Qt.Unchecked)
                item.setData(0, Qt.UserRole, name)
                item.setToolTip(0, self._tooltip(row))
                self._write_badges(item, name)
                self._tree.addTopLevelItem(item)
        finally:
            self._suppress = False
        self._sync_checks()

    def _tooltip(self, row: dict) -> str:
        return (f"{row['name']}\n{row['n_layers']} layers · "
                f"{row['total_depth']:.1f} m"
                + (f" · Vs30 {row['vs30']:.0f} m/s" if row.get("vs30") else ""))

    def _write_badges(self, item: QTreeWidgetItem, name: str) -> None:
        for i, (kind, _h, tool) in enumerate(self._badge_cols, start=1):
            if kind == "sett":
                assigned = self._app.has_settings(tool, name)
                item.setData(i, Qt.UserRole, assigned)
                item.setToolTip(i, "Settings assigned" if assigned
                                else "Default settings")
            else:
                status = self._app.profile_status(tool, name)
                item.setData(i, Qt.UserRole, status)
                item.setToolTip(i, f"{tool.label}: {_status_label(status)}")

    def _find_item(self, name: str) -> Optional[QTreeWidgetItem]:
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == name:
                return item
        return None

    def _refresh_row(self, name: str) -> None:
        item = self._find_item(name)
        if item is not None:
            self._write_badges(item, name)

    # ==================================================================
    #  AppState → panel
    # ==================================================================
    def _on_status_changed(self, tool, name: str) -> None:
        self._refresh_row(str(name))

    def _on_focus_changed(self, name) -> None:
        item = self._find_item(name) if name else None
        self._suppress = True
        try:
            self._tree.setCurrentItem(item)
        finally:
            self._suppress = False

    def _sync_checks(self) -> None:
        checked = set(self._app.checked_profiles())
        total = self._tree.topLevelItemCount()
        self._suppress = True
        try:
            for i in range(total):
                item = self._tree.topLevelItem(i)
                name = item.data(0, Qt.UserRole)
                item.setCheckState(
                    0, Qt.Checked if name in checked else Qt.Unchecked)
        finally:
            self._suppress = False
        n = len(checked)
        self._count_lbl.setText(f"{n}/{total} checked" if total else "empty")
        self._select_all.blockSignals(True)
        self._select_all.setCheckState(
            Qt.Unchecked if n == 0 else
            Qt.Checked if n == total else Qt.PartiallyChecked)
        self._select_all.blockSignals(False)
        self._update_assign_btn()

    # ==================================================================
    #  Panel → AppState
    # ==================================================================
    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._suppress or column != 0:
            return
        name = item.data(0, Qt.UserRole)
        self._app.set_profile_checked(
            name, item.checkState(0) == Qt.Checked)

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if not self._suppress:
            self._app.set_focus(item.data(0, Qt.UserRole))

    def _on_select_all(self) -> None:
        state = self._select_all.checkState()
        if state == Qt.PartiallyChecked:      # cycle → all
            self._select_all.setCheckState(Qt.Checked)
            state = Qt.Checked
        names = self._app.profile_names() if state == Qt.Checked else []
        self._app.set_checked(names)

    def _on_remove(self) -> None:
        targets = self._assign_targets()
        for name in targets:
            self._app.remove_profile(name)

    def _assign_targets(self) -> List[str]:
        checked = self._app.checked_profiles()
        if checked:
            return checked
        return [self._app.focus] if self._app.focus else []

    def _update_assign_btn(self) -> None:
        targets = self._assign_targets()
        if not targets:
            self._assign_btn.setText("Assign settings")
            self._assign_btn.setEnabled(False)
        elif len(targets) == 1:
            self._assign_btn.setText(f"Assign settings to {targets[0]}")
            self._assign_btn.setEnabled(True)
        else:
            self._assign_btn.setText(
                f"Assign settings to {len(targets)} checked")
            self._assign_btn.setEnabled(True)

    def _on_assign(self) -> None:
        targets = self._assign_targets()
        if targets:
            self.assign_settings_requested.emit(targets)

    def _on_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        name = item.data(0, Qt.UserRole)
        menu = QMenu(self)
        focus_act = menu.addAction("Focus")
        remove_act = menu.addAction("Remove")
        chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if chosen is focus_act:
            self._app.set_focus(name)
        elif chosen is remove_act:
            self._app.remove_profile(name)


__all__ = ["ProfilesPanel"]
