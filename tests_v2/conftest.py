"""tests_v2 conftest — the REBUILD's suite (PySide6-offscreen lane).

Runs as a SEPARATE pytest process from the legacy ``tests/`` (whose conftest
force-imports PyQt5).  Never import PyQt5 here.  api tests are Qt-free;
matplotlib is pinned to the headless Agg backend so no test accidentally
pulls a Qt binding through pyplot.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["QT_API"] = "pyside6"

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
