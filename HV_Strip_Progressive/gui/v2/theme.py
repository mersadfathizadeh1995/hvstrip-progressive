"""Tri-palette **violet** theme for HV Invert, on ``theme_core``.

Stage-5 (house-style uplift) replacement of the old two-string light/dark
QSS.  It follows the family pattern proven in :mod:`bedrock_mapping.gui.theme`
(the copy-source): three concrete
:class:`~hvsr_pro.packages.theme_core.Palette` constants (LIGHT / GRAY / DARK)
with a **violet accent** — HV Invert's identity colour, distinct from HV Hub's
brand red (``#C8102E``), gui_v2's analysis blue (``#0078D4``) and bedrock's
forest green (``#3E7A45``).  ``primary`` aliases ``accent``.

Theming is **per-window** and the app **follows**
``theme_core.theme_authority`` (HV Hub is the sole mode writer): no theme
picker here, no persistence here.  :func:`apply_theme` never touches the
process-global stylesheet unless the *target* itself is the ``QApplication``
(the legacy ``app.py`` still calls ``apply_theme(app, "light")`` — that path
also seeds the pyqtgraph background/foreground as an **app baseline** so the
old GUI keeps working unchanged until Stage 06 retires it).

Back-compat surface (do NOT break — the legacy GUI depends on it):

* ``apply_theme(app, "light" | "dark")`` — the legacy signature is a subset of
  the new ``apply_theme(target, mode)`` (``"light"``/``"dark"`` are valid
  modes), so no legacy call site needs to change.
* ``DARK_THEME`` / ``LIGHT_THEME`` / ``DEFAULT_THEME`` constants are kept as
  thin aliases so any residual import still resolves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication, QWidget

from hvsr_pro.packages import theme_core as _core
from hvsr_pro.packages.theme_core import (
    Palette,
    THEME_MODE_LIGHT,
    seed_qpalette,
)
from hvsr_pro.packages.theme_core.resolve import ThemeInput

#: Vendored checkmark glyph for checkbox/tree indicators (so a checked box
#: shows a tick, not an accent-filled square).  Forward slashes for QSS url().
_CHECK_SVG = (Path(__file__).resolve().parent / "assets" / "check.svg").as_posix()
#: White variant — drawn over the accent-filled layer-tree indicator.
_CHECK_WHITE_SVG = (
    Path(__file__).resolve().parent / "assets" / "check_white.svg"
).as_posix()


# ----------------------------------------------------------------------
# Amber palettes (neutral surfaces shared with the family; only the accent
# family carries the HV Invert identity)
# ----------------------------------------------------------------------

#: Light palette — default.  Accent ``#C87A20``: a balanced violet/purple.
LIGHT = Palette(
    name="light",
    bg="#F5F5F5",
    card="#FFFFFF",
    card_alt="#FAFAFA",
    header="#F0F0F0",
    border="#D0D0D0",
    border_strong="#C0C0C0",
    fg="#1E1E1E",
    muted="#666666",
    disabled="#999999",
    primary="#C87A20",
    primary_fg="#FFFFFF",
    accent="#C87A20",
    accent_hover="#D98F33",
    accent_pressed="#A5641A",
    accent_light="#F0DCC3",
    accent_vlight="#F9EFE2",
    success="#27AE60",
    warning="#F0A020",
    danger="#E63946",
)

#: Gray palette — mid-tone slate surfaces (gui_v2 GRAY family); the violet is
#: darkened so it keeps contrast against the lighter ``#C8C8CE`` bg, and
#: ``accent_vlight`` is bumped so an active-step fill stays distinguishable.
GRAY = Palette(
    name="gray",
    bg="#C8C8CE",
    card="#D8D8DE",
    card_alt="#CECED4",
    header="#B8B8BE",
    border="#9090A0",
    border_strong="#7A7A82",
    fg="#1A1A1E",
    muted="#3A3A3E",
    disabled="#7A7A7E",
    primary="#A5641A",
    primary_fg="#FFFFFF",
    accent="#A5641A",
    accent_hover="#C87A20",
    accent_pressed="#7E4C13",
    accent_light="#DCC5A5",
    accent_vlight="#CDB28C",
    success="#1C8E51",
    warning="#CC8800",
    danger="#C62E3A",
)

#: Dark palette — dark surfaces (gui_v2 DARK family); the violet is brightened
#: for legibility against ``#1E1E22`` and ``accent_vlight`` is a deep amber
#: fill that reads against the ``#2A2A30`` card.
DARK = Palette(
    name="dark",
    bg="#1E1E22",
    card="#2A2A30",
    card_alt="#26262C",
    header="#16161A",
    border="#3A3A44",
    border_strong="#4A4A54",
    fg="#F0F0F3",
    muted="#A8A8B0",
    disabled="#6A6A72",
    primary="#E09A45",
    primary_fg="#FFFFFF",
    accent="#E09A45",
    accent_hover="#EBAF63",
    accent_pressed="#C87A20",
    accent_light="#43321F",
    accent_vlight="#53401F",
    success="#34C876",
    warning="#F5B73A",
    danger="#EF4D5C",
)

#: ``{name: palette}`` lookup used by :func:`resolve_palette`.
PALETTES: dict[str, Palette] = {
    LIGHT.name: LIGHT,
    GRAY.name: GRAY,
    DARK.name: DARK,
}


def resolve_palette(
    theme: ThemeInput = THEME_MODE_LIGHT,
    app: Optional[QApplication] = None,
) -> Palette:
    """Resolve a mode string / palette / ``None`` into a concrete palette.

    Delegates to the shared :func:`theme_core.resolve_palette` against this
    package's :data:`PALETTES` (``system`` resolves via Qt's colour scheme).
    """
    return _core.resolve_palette(theme, PALETTES, app)


# ----------------------------------------------------------------------
# QSS
# ----------------------------------------------------------------------


def build_qss(p: Palette) -> str:
    """Build the HV Invert role-QSS from *p*.

    Deterministic and side-effect-free.  Role/dynamic-property rules only —
    no hard-coded hex; every colour is a palette token.  (Copied from the
    bedrock reference and kept in step with it: stage-ribbon step buttons,
    header strips, role labels, buttons incl. ``primary``, inputs,
    checkboxes with the vendored tick, lists/tables, progress bar,
    collapsible headers, side-rail chrome, splitter, canvas tabs, tooltip.)
    """
    return f"""
QMainWindow, QDialog {{ background-color: {p.bg}; }}
QWidget {{ background-color: {p.card}; color: {p.fg}; font-size: 12px; }}
QFrame {{ background-color: transparent; }}

/* -- Menu / status chrome ------------------------------------ */
QMenuBar {{
    background-color: {p.header}; color: {p.fg};
    border-bottom: 1px solid {p.border};
}}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background-color: {p.accent_light}; }}
QMenu {{
    background-color: {p.card}; color: {p.fg};
    border: 1px solid {p.border};
}}
QMenu::item {{ padding: 5px 22px; }}
QMenu::item:selected {{ background-color: {p.accent_light}; }}
QStatusBar {{
    background-color: {p.header}; color: {p.fg};
    border-top: 1px solid {p.border}; min-height: 22px;
}}
QStatusBar::item {{ border: 0; }}

/* -- Header / cards ------------------------------------------ */
QFrame[role="header"] {{
    background-color: {p.header}; border-bottom: 1px solid {p.border};
}}
QFrame[role="brand"] {{ background-color: {p.accent}; border: 0; }}
QFrame[role="brand"] QLabel {{ background-color: transparent; color: {p.primary_fg}; }}
QFrame[role="card"] {{
    background-color: {p.card}; border: 1px solid {p.border};
    border-radius: 4px;
}}
QFrame[role="hline"] {{ background: {p.border}; max-height: 1px; min-height: 1px; }}

/* -- Labels --------------------------------------------------- */
QLabel {{ background: transparent; color: {p.fg}; }}
QLabel[role="muted"] {{ color: {p.muted}; }}
QLabel[role="title"] {{ font-weight: 600; }}
QLabel[role="h1"] {{ font-size: 22px; font-weight: 300; }}
QLabel[role="h2"] {{ font-size: 14px; font-weight: 600; }}
QLabel[role="heading"] {{ color: {p.fg}; font-size: 13px; font-weight: 600; }}
QLabel[role="caption"] {{ color: {p.muted}; font-size: 11px; font-style: italic; }}
QLabel[role="ok"] {{ color: {p.success}; font-weight: 600; }}
QLabel[role="error"] {{ color: {p.danger}; font-weight: 600; }}
QLabel[role="warning"] {{ color: {p.warning}; font-weight: 600; }}

/* -- Buttons (gui_v2 density — compact, not chunky) ----------- */
QPushButton {{
    background-color: {p.card}; color: {p.fg};
    border: 1px solid {p.border_strong}; border-radius: 3px;
    padding: 3px 10px; min-height: 18px;
}}
QPushButton:hover {{ background-color: {p.accent_vlight}; border-color: {p.accent}; }}
QPushButton:pressed {{ background-color: {p.accent_light}; }}
QPushButton:disabled {{
    color: {p.disabled}; background-color: {p.card_alt};
    border-color: {p.border};
}}
QPushButton[primary="true"] {{
    background-color: {p.accent}; color: {p.primary_fg};
    border: 1px solid {p.accent}; font-weight: 600;
}}
QPushButton[primary="true"]:hover {{
    background-color: {p.accent_hover}; border-color: {p.accent_hover};
}}
QPushButton[primary="true"]:pressed {{
    background-color: {p.accent_pressed}; border-color: {p.accent_pressed};
}}

/* -- Stage ribbon step buttons (gui_v2 idiom) ----------------- */
QPushButton[step] {{ background-color: transparent; }}
QPushButton[step="active"] {{
    text-align: left; padding: 6px 14px;
    border: 0;
    border-left: 3px solid {p.accent};
    border-bottom: 2px solid {p.accent};
    background-color: {p.accent_vlight}; color: {p.fg};
    font-weight: 600;
}}
QPushButton[step="active"]:hover {{ background-color: {p.accent_vlight}; }}
QPushButton[step="done"] {{
    text-align: left; padding: 6px 14px;
    border: 0; border-left: 3px solid {p.success};
    background-color: transparent; color: {p.fg};
}}
QPushButton[step="done"]:hover {{ background-color: {p.card_alt}; }}
QPushButton[step="ready"] {{
    text-align: left; padding: 6px 14px;
    border: 0; border-left: 3px solid {p.warning};
    background-color: transparent; color: {p.fg};
}}
QPushButton[step="ready"]:hover {{ background-color: {p.card_alt}; }}
QPushButton[step="running"] {{
    text-align: left; padding: 6px 14px;
    border: 0; border-left: 3px solid {p.accent};
    background-color: {p.accent_vlight}; color: {p.fg};
}}
QPushButton[step="error"] {{
    text-align: left; padding: 6px 14px;
    border: 0; border-left: 3px solid {p.danger};
    background-color: transparent; color: {p.fg};
}}
QPushButton[step="error"]:hover {{ background-color: {p.card_alt}; }}
QPushButton[step="locked"] {{
    text-align: left; padding: 6px 14px;
    border: 0; border-left: 3px solid transparent;
    background-color: transparent; color: {p.disabled};
}}

/* -- Inputs ---------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {p.card}; color: {p.fg};
    border: 1px solid {p.border_strong}; border-radius: 3px;
    padding: 3px 6px; selection-background-color: {p.accent_light};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {p.accent};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{
    color: {p.disabled}; background-color: {p.card_alt};
}}
QComboBox::drop-down {{ border: 0; width: 18px; }}

/* -- Checkboxes / tree-item ticks (a checkmark, not a filled box) --- */
QCheckBox {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator,
QTreeView::indicator, QTreeWidget::indicator,
QTableView::indicator, QTableWidget::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {p.border_strong}; border-radius: 3px;
    background-color: {p.card};
}}
QCheckBox::indicator:hover,
QTreeView::indicator:hover, QTreeWidget::indicator:hover {{
    border-color: {p.accent};
}}
QCheckBox::indicator:checked,
QTreeView::indicator:checked, QTreeWidget::indicator:checked,
QTableView::indicator:checked, QTableWidget::indicator:checked {{
    border-color: {p.accent};
    image: url("{_CHECK_SVG}");
}}
QCheckBox::indicator:indeterminate {{ background-color: {p.accent_light}; }}
QCheckBox::indicator:disabled,
QTreeView::indicator:disabled, QTreeWidget::indicator:disabled {{
    background-color: {p.card_alt}; border-color: {p.border};
}}

/* -- Lists / tables baseline ----------------------------------- */
QListWidget, QTreeView, QTableView {{
    background-color: {p.card}; color: {p.fg};
    border: 1px solid {p.border}; border-radius: 3px;
    alternate-background-color: {p.card_alt};
    selection-background-color: {p.accent_light};
    selection-color: {p.fg};
}}
QListWidget::item {{ padding: 5px 8px; }}
QListWidget::item:selected {{
    background-color: {p.accent_light}; color: {p.fg};
}}
QHeaderView::section {{
    background-color: {p.header}; color: {p.fg};
    border: 0; border-bottom: 1px solid {p.border};
    border-right: 1px solid {p.border};
    padding: 4px 8px; font-weight: 600;
}}

/* -- Progress / scroll ------------------------------------------ */
QProgressBar {{
    background-color: {p.card_alt};
    border: 1px solid {p.border}; border-radius: 3px;
    max-height: 12px; text-align: center; color: {p.fg};
}}
QProgressBar::chunk {{ background-color: {p.accent}; border-radius: 2px; }}
QScrollBar:vertical {{
    background: {p.bg}; width: 11px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {p.border_strong}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.accent}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {p.bg}; height: 11px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {p.border_strong}; border-radius: 5px; min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* -- Group boxes ------------------------------------------------ */
QGroupBox {{
    border: 1px solid {p.border}; border-radius: 4px;
    margin-top: 10px; padding-top: 6px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px; padding: 0 4px;
    color: {p.fg};
}}

/* -- Collapsible group header (CollapsibleGroup) — violet strip when open --- */
QToolButton[role="groupHeader"] {{
    background: transparent; color: {p.fg};
    border: 0; border-left: 3px solid transparent; border-radius: 3px;
    padding: 5px 8px; font-weight: 600; text-align: left;
}}
QToolButton[role="groupHeader"]:checked {{
    background-color: {p.accent_vlight}; border-left: 3px solid {p.accent};
}}
QToolButton[role="groupHeader"]:hover {{ background-color: {p.accent_light}; color: {p.fg}; }}
QToolButton[role="groupHeader"]::menu-indicator {{ image: none; }}

/* -- Sub-stage tabs (SubBreadcrumb) — underline on the active tab --- */
QFrame[role="subBar"] {{
    background-color: {p.header}; border-bottom: 1px solid {p.border};
}}
QToolButton[role="subTab"] {{
    background: transparent; color: {p.muted};
    border: 0; border-bottom: 2px solid transparent;
    padding: 4px 7px; font-weight: 600;
}}
QToolButton[role="subTab"]:hover {{ color: {p.accent}; }}
QToolButton[role="subTab"]:checked {{
    color: {p.accent}; border-bottom: 2px solid {p.accent};
}}
QToolButton[role="subTab"]:disabled {{ color: {p.disabled}; }}

/* -- Workbench collapse rails (CollapsibleSideRail) ----------- */
QWidget[role="railStrip"] {{ background-color: {p.header}; }}
QToolButton[role="railChevron"] {{
    background: transparent; color: {p.muted};
    border: 0; border-radius: 3px; font-size: 12px; padding: 0;
}}
QToolButton[role="railChevron"]:hover {{
    background-color: {p.accent_vlight}; color: {p.accent};
}}

/* -- Splitter handles ----------------------------------------- */
QSplitter::handle {{ background-color: {p.border}; }}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:vertical {{ height: 4px; }}
QSplitter::handle:hover {{ background-color: {p.accent}; }}

/* -- Canvas tab strip (CanvasFrame) + bottom dock tabs -------- */
QTabWidget#CanvasFrame::pane, QTabWidget#WorkbenchDock::pane {{
    border: 1px solid {p.border}; border-radius: 4px;
    background-color: {p.card};
}}
QTabBar::tab {{
    background-color: {p.header}; color: {p.muted};
    border: 1px solid {p.border}; border-bottom: 0;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
    padding: 6px 16px; margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {p.card}; color: {p.accent};
    font-weight: 600; border-bottom: 2px solid {p.accent};
}}
QTabBar::tab:hover {{ color: {p.fg}; }}

/* -- Layer tree (professional/minimal: flat, roomy rows, violet
      accent-filled check indicators with a white tick) -------------- */
QTreeWidget[role="layerTree"] {{
    background: transparent; color: {p.fg};
    border: none; outline: 0;
}}
QTreeWidget[role="layerTree"]::item {{
    padding: 5px 4px; min-height: 24px; border-radius: 4px;
}}
QTreeWidget[role="layerTree"]::item:hover {{
    background-color: {p.accent_vlight};
}}
QTreeWidget[role="layerTree"]::item:selected {{
    background-color: {p.accent_light}; color: {p.fg};
}}
QTreeWidget[role="layerTree"]::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {p.border_strong}; border-radius: 4px;
    background-color: {p.card};
}}
QTreeWidget[role="layerTree"]::indicator:hover {{ border-color: {p.accent}; }}
QTreeWidget[role="layerTree"]::indicator:checked {{
    background-color: {p.accent}; border-color: {p.accent};
    image: url("{_CHECK_WHITE_SVG}");
}}
QTreeWidget[role="layerTree"] QHeaderView::section {{
    background: transparent; color: {p.muted};
    border: none; border-bottom: 1px solid {p.border};
    padding: 3px 4px; font-weight: 600;
}}

/* -- Tooltip (fixed dark, same convention as the family) ------ */
QToolTip {{
    background-color: #2d2d2d; color: #ffffff;
    border: 1px solid #1a1a1a; padding: 6px 8px; font-size: 12px;
}}
"""


# ----------------------------------------------------------------------
# Apply
# ----------------------------------------------------------------------


def _app_of(target: object) -> Optional[QApplication]:
    if isinstance(target, QApplication):
        return target
    return QApplication.instance()


def _configure_pyqtgraph(palette: Palette) -> None:
    """App baseline only: keep pyqtgraph's default bg/fg in step with the
    mode so any live canvas built before Stage 06 (or the legacy GUI) reads
    on the right surface.  Best-effort — pyqtgraph may not be importable in a
    headless unit test, and per-window canvases re-theme themselves (T028)."""
    try:
        import pyqtgraph as pg

        pg.setConfigOptions(
            antialias=True, background=palette.bg, foreground=palette.fg,
        )
    except Exception:  # noqa: BLE001 — a baseline convenience, never fatal
        pass


def apply_theme(
    target: "QApplication | QWidget",
    mode: ThemeInput = THEME_MODE_LIGHT,
) -> Palette:
    """Apply *mode* to *target* (a window or the app) and return the palette.

    Per-window theming (the family convention): the QSS + seeded
    :class:`QPalette` are set on *target* only — a window-level stylesheet
    overrides the app baseline within that window's subtree, so the Hub and an
    HV Invert window keep their own chrome simultaneously and focus changes
    never re-style anything.  The ``pg.setConfigOptions`` app baseline is
    applied **only** when *target* is the ``QApplication`` (the legacy
    ``app.py`` path); per-window calls set QSS + QPalette only.
    """
    palette = resolve_palette(mode, _app_of(target))
    target.setPalette(seed_qpalette(palette))
    target.setStyleSheet(build_qss(palette))
    if isinstance(target, QApplication):
        _configure_pyqtgraph(palette)
    return palette


# ----------------------------------------------------------------------
# Back-compat aliases (legacy import surface — retired in Stage 06)
# ----------------------------------------------------------------------
#: The old two-string API exposed module-level QSS constants + a default mode.
#: A couple of residual imports may still reference them; keep thin aliases so
#: nothing breaks before Stage 06 removes the legacy GUI.
DEFAULT_THEME = "light"
LIGHT_THEME = build_qss(LIGHT)
DARK_THEME = build_qss(DARK)


__all__ = [
    "LIGHT",
    "GRAY",
    "DARK",
    "PALETTES",
    "Palette",
    "resolve_palette",
    "build_qss",
    "apply_theme",
    "DEFAULT_THEME",
    "LIGHT_THEME",
    "DARK_THEME",
]
