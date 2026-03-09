"""Constants shared across the All Profiles View modules."""

# Color palette names (custom + matplotlib colormaps)
PALETTES = [
    "Classic", "Bold", "Earth", "Nordic", "Sunset",
    "Green", "Blue", "Orange", "Red", "Purple",
    "tab10", "tab20", "Set1", "Set2", "Set3",
    "Pastel1", "Paired", "Dark2", "Accent",
]

# Custom color schemes keyed by lowercase palette name
BUILTIN_COLORS = {
    "classic": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"],
    "bold": ["#e6194b", "#3cb44b", "#ffe119", "#4363d8",
             "#f58231", "#911eb4", "#42d4f4", "#f032e6"],
    "earth": ["#8B4513", "#228B22", "#DAA520", "#CD853F",
              "#2E8B57", "#D2691E", "#6B8E23", "#A0522D"],
    "nordic": ["#2E4057", "#048A81", "#54C6EB", "#8EE3EF",
               "#F25C54", "#F4845F", "#F7B267", "#7D82B8"],
    "sunset": ["#FF6B6B", "#FFA07A", "#FFD700", "#FF8C00",
               "#FF4500", "#DC143C", "#FF69B4", "#FF1493"],
    "green": ["#2ca02c", "#228B22", "#006400", "#32CD32",
              "#66CDAA", "#3CB371", "#00FA9A", "#90EE90"],
    "blue": ["#1f77b4", "#4169E1", "#000080", "#4682B4",
             "#6495ED", "#00BFFF", "#87CEEB", "#1E90FF"],
    "orange": ["#ff7f0e", "#FF8C00", "#FFA500", "#FF6347",
               "#E9967A", "#FFD700", "#F4A460", "#FF4500"],
    "red": ["#d62728", "#DC143C", "#B22222", "#FF0000",
            "#CD5C5C", "#FF6B6B", "#8B0000", "#E74C3C"],
    "purple": ["#9467bd", "#800080", "#8B008B", "#9932CC",
               "#BA55D3", "#DDA0DD", "#7B68EE", "#6A0DAD"],
}

# Unicode → matplotlib marker code mapping
MARKER_SHAPES = {
    "★": "*", "◆": "D", "●": "o", "▲": "^",
    "▼": "v", "■": "s", "+": "P", "✕": "X",
}

# Y-limit auto-scaling methods
Y_LIMIT_METHODS = ["Auto", "95th Percentile", "Mean + 3σ", "Mean + 2×IQR"]

# Pre-defined figure dimensions (width, height) in inches
FIGURE_SIZES = {
    "Standard (10×7)": (10, 7),
    "Large (14×10)": (14, 10),
    "Publication (12×8)": (12, 8),
    "Wide (16×6)": (16, 6),
}
