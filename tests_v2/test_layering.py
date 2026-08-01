"""The AST layering guard — Core-API-Consumers, adapted from invert_hvsr's.

REPORT-ONLY until the house-style cutover (``ENFORCING = False``): the rules
that already hold (core/api Qt-free, api never imports gui, the package
import is Qt-free) are ENFORCED now; the rules the legacy PyQt5 GUI still
violates (PySide6-only, no consumer→frozen-compute) are collected and
REPORTED, and flip to enforcing when ``gui/v2`` replaces the legacy tree.

Rules:
  a. nothing under ``core/`` or ``api/`` imports Qt, ``gui``, or ``app``;
  b. ``api/`` never imports ``gui``;
  c. ``import HV_Strip_Progressive`` is Qt-free (subprocess probe);
  d. PySide6-only — PyQt5/PyQt6 banned (legacy ``gui/`` exempt until cutover);
  e. no consumer imports the FROZEN core compute — the api is the only path
     (legacy ``gui/`` exempt until cutover; ``gui/v2`` enforced from day one).
"""

from __future__ import annotations

import ast
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

#: Flip to True at the house-style cutover (legacy gui retired).
ENFORCING = False

PKG_NAME = "HV_Strip_Progressive"
DIST_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = DIST_ROOT / PKG_NAME

QT_TOP_LEVEL = {"PySide6", "PyQt5", "PyQt6", "qtpy", "shiboken6", "shiboken2"}
PYQT_TOP_LEVEL = {"PyQt5", "PyQt6"}

#: The frozen compute: reachable ONLY through the api facade.  The shared
#: data model (``soil_profile``, ``velocity_utils``, ``vs_average``) and the
#: analysis modules (``peak_detection``, ``report_generator``,
#: ``dual_resonance``, ``advanced_analysis``) stay directly importable.
CORE_COMPUTE = {
    "stripper",
    "hv_forward",
    "batch_workflow",
    "engines",
    "hv_postprocess",
}

#: Paths exempt from rules (d) and (e) until the cutover.
_LEGACY_GUI_PREFIXES = ("gui",)          # the whole legacy tree pre-v2
_V2_PREFIX = ("gui", "v2")               # enforced from day one


def _iter_py_files() -> Iterator[Tuple[Path, Path]]:
    for py in sorted(PKG_ROOT.rglob("*.py")):
        rel = py.relative_to(PKG_ROOT)
        if "__pycache__" in rel.parts:
            continue
        yield py, rel


def _module_package(rel: Path) -> str:
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    else:
        parts = parts[:-1]
    return ".".join([PKG_NAME, *parts]) if parts else PKG_NAME


def _imports_of(py: Path, rel: Path) -> List[str]:
    tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
    pkg = _module_package(rel)
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                out.append(node.module or "")
            else:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)]
                target = ".".join(base)
                if node.module:
                    target = f"{target}.{node.module}" if target else node.module
                out.append(target)
    return out


def _is_legacy_gui(rel: Path) -> bool:
    parts = rel.parts
    return (parts[: len(_LEGACY_GUI_PREFIXES)] == _LEGACY_GUI_PREFIXES
            and parts[: len(_V2_PREFIX)] != _V2_PREFIX)


def _report_or_fail(violations: List[str], rule: str) -> None:
    if not violations:
        return
    msg = f"[layering:{rule}] " + "; ".join(violations)
    if ENFORCING:
        pytest.fail(msg)
    warnings.warn("REPORT-ONLY " + msg, stacklevel=2)


# ----------------------------------------------------------------------
#  (a) + (b) — enforced NOW (they already hold)
# ----------------------------------------------------------------------
def test_core_and_api_are_qt_free_and_gui_free():
    bad: List[str] = []
    for py, rel in _iter_py_files():
        layer = rel.parts[0] if rel.parts else ""
        if layer not in ("core", "api"):
            continue
        for imp in _imports_of(py, rel):
            top = imp.split(".")[0]
            if top in QT_TOP_LEVEL:
                bad.append(f"{rel}: imports {imp}")
            if imp.startswith(f"{PKG_NAME}.gui") or imp == f"{PKG_NAME}.app":
                bad.append(f"{rel}: imports {imp}")
    assert not bad, "core/api must stay Qt-free and gui-free: " + "; ".join(bad)


def test_api_never_imports_gui():
    bad: List[str] = []
    for py, rel in _iter_py_files():
        if rel.parts[0] != "api":
            continue
        for imp in _imports_of(py, rel):
            if imp.startswith(f"{PKG_NAME}.gui"):
                bad.append(f"{rel}: imports {imp}")
    assert not bad, "; ".join(bad)


# ----------------------------------------------------------------------
#  (c) — enforced NOW
# ----------------------------------------------------------------------
def test_package_import_is_qt_free():
    code = (
        "import sys\n"
        f"import {PKG_NAME}\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in "
        f"{sorted(QT_TOP_LEVEL)!r}]\n"
        "raise SystemExit(1 if loaded else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(DIST_ROOT),
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"`import {PKG_NAME}` pulled in Qt.\n{proc.stderr}"
    )


# ----------------------------------------------------------------------
#  (d) + (e) — report-only until the cutover; gui/v2 enforced always
# ----------------------------------------------------------------------
def test_pyside6_only():
    legacy: List[str] = []
    v2: List[str] = []
    for py, rel in _iter_py_files():
        for imp in _imports_of(py, rel):
            if imp.split(".")[0] in PYQT_TOP_LEVEL:
                (legacy if _is_legacy_gui(rel) else v2).append(
                    f"{rel}: imports {imp}")
    # Non-legacy PyQt is ALWAYS a failure (v2 / core / api / research).
    assert not v2, "PyQt banned outside the retiring legacy gui: " + "; ".join(v2)
    _report_or_fail(legacy, "pyside6-only(legacy gui)")


def test_no_consumer_imports_frozen_compute():
    legacy: List[str] = []
    v2: List[str] = []
    frozen = {f"{PKG_NAME}.core.{m}" for m in CORE_COMPUTE}
    for py, rel in _iter_py_files():
        if rel.parts[0] not in ("gui",):
            continue
        for imp in _imports_of(py, rel):
            if any(imp == f or imp.startswith(f + ".") for f in frozen):
                (legacy if _is_legacy_gui(rel) else v2).append(
                    f"{rel}: imports {imp}")
    assert not v2, (
        "gui/v2 must reach compute ONLY via the api facade: " + "; ".join(v2))
    _report_or_fail(legacy, "frozen-compute(legacy gui)")
