"""
Style Constants
===============

Centralized styling constants for the HV Strip Progressive GUI.
Follows bedrock_mapping visual language: emoji headers, tight margins,
monospace previews, gray secondary text.
"""

# ── Margins & Spacing ───────────────────────────────────────────
OUTER_MARGINS = (4, 4, 4, 4)     # Main window / top-level containers
INNER_MARGINS = (6, 6, 6, 6)     # Widget content areas
COLLAPSIBLE_MARGINS = (8, 4, 4, 4)  # Inside collapsible sections

# ── Splitter ────────────────────────────────────────────────────
LEFT_PANEL_MIN = 350
LEFT_PANEL_MAX = 550
INITIAL_SPLITTER_SIZES = [400, 1000]

# ── CSS Snippets ────────────────────────────────────────────────
SECONDARY_LABEL = "color: #555; font-size: 11px;"
MONOSPACE_PREVIEW = (
    "font-family: Consolas, 'Courier New', monospace; "
    "font-size: 11px; background: #fafafa; border: 1px solid #ddd; "
    "padding: 4px;"
)
STATUS_OK = "color: green; font-weight: bold;"
STATUS_ERR = "color: red; font-weight: bold;"
BUTTON_PRIMARY = (
    "QPushButton { background-color: #2E86AB; color: white; "
    "padding: 6px 16px; border-radius: 4px; font-weight: bold; } "
    "QPushButton:hover { background-color: #256E8D; } "
    "QPushButton:disabled { background-color: #aaa; }"
)
BUTTON_SUCCESS = (
    "QPushButton { background-color: #27AE60; color: white; "
    "padding: 6px 16px; border-radius: 4px; font-weight: bold; } "
    "QPushButton:hover { background-color: #219A52; } "
    "QPushButton:disabled { background-color: #aaa; }"
)
BUTTON_DANGER = (
    "QPushButton { background-color: #E63946; color: white; "
    "padding: 6px 16px; border-radius: 4px; font-weight: bold; } "
    "QPushButton:hover { background-color: #C62E3A; } "
    "QPushButton:disabled { background-color: #aaa; }"
)
GEAR_BUTTON = (
    "QPushButton { border: none; padding: 2px; } "
    "QPushButton:hover { background-color: #e0e0e0; border-radius: 3px; }"
)

# ── Emoji Prefixes (for group titles) ──────────────────────────
EMOJI = {
    "home": "🏠",
    "forward": "📊",
    "figures": "📈",
    "settings": "⚙",
    "file": "📁",
    "folder": "📂",
    "config": "🔧",
    "engine": "🔬",
    "frequency": "📡",
    "peak": "📍",
    "plot": "🎨",
    "chart": "📉",
    "save": "💾",
    "export": "📤",
    "run": "▶",
    "stop": "⏹",
    "ok": "✅",
    "error": "❌",
    "warning": "⚠",
    "info": "ℹ",
    "layer": "🪨",
    "profile": "📐",
    "batch": "📋",
    "dual": "🔀",
    "report": "📄",
    "interactive": "👆",
    "research": "🔬",
    "study": "📝",
    "metrics": "📏",
    "generate": "🔄",
    "validate": "✔",
    "field": "🌍",
    "latex": "📜",
}
