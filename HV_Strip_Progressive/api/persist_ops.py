"""Picked-peaks persistence + results-folder rehydration (spec 002, T2-S1).

The legacy app's post-Finish write-back chain, relocated behind the facade
(Qt-free) and extended:

* the step ``*summary*.csv`` peak cells are updated (or a minimal summary
  created) EXACTLY as the legacy ``HVStripWindow._update_summary_csv`` /
  ``_create_minimal_summary_csv`` did — the frozen report generator reads
  ``Peak_Frequency_Hz`` / ``Peak_Amplitude`` from these files;
* ``vs_results.json`` next to the step folders carries the per-step
  Vs30/VsAvg/bedrock context (legacy-identical shape);
* NEW: a ``picked_peaks.json`` sidecar carries the FULL picks — f0 **and
  secondaries** (the legacy app silently dropped secondaries from
  persistence; spec 002 DC-8 fixes that), label positions, and the Vs
  context — so re-opening a results folder restores everything.

Rehydration (:func:`rehydrate_results_folder`) rebuilds a
:class:`~.strip_engine.StripResult` from a results tree WITHOUT recompute,
reading both v2 trees (sidecar present) and legacy trees (summaries +
``vs_results.json`` only).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.batch_workflow import find_step_folders
from .forward_engine import PeakInfo
from .strip_engine import StepResult, StripResult

logger = logging.getLogger(__name__)

#: The v2 full-fidelity picks sidecar, written next to the step folders.
SIDECAR_NAME = "picked_peaks.json"
SIDECAR_VERSION = 1

#: The legacy Vs-context sidecar (still written for the frozen reporter,
#: still read for legacy trees).
VS_RESULTS_NAME = "vs_results.json"


# ---------------------------------------------------------------------------
# Peak (de)serialisation
# ---------------------------------------------------------------------------


def peak_to_dict(p: PeakInfo) -> Dict[str, Any]:
    return {
        "frequency": p.frequency,
        "amplitude": p.amplitude,
        "index": p.index,
        "label": p.label,
        "source": p.source,
        "label_pos": list(p.label_pos) if p.label_pos else None,
    }


def peak_from_dict(d: Dict[str, Any]) -> PeakInfo:
    return PeakInfo(
        frequency=float(d.get("frequency", 0.0)),
        amplitude=float(d.get("amplitude", 0.0)),
        index=int(d.get("index", 0)),
        label=str(d.get("label", "")),
        source=str(d.get("source", "manual")),
        label_pos=(list(d["label_pos"])
                   if d.get("label_pos") is not None else None),
    )


def _normalize_step_picks(pdata: Dict[str, Any]) -> Dict[str, Any]:
    """One step's picks → the canonical sidecar shape (plain dicts)."""
    out: Dict[str, Any] = {}
    f0 = pdata.get("f0")
    if isinstance(f0, PeakInfo):
        out["f0"] = peak_to_dict(f0)
    elif isinstance(f0, dict):
        out["f0"] = peak_to_dict(peak_from_dict(f0))
    elif isinstance(f0, (tuple, list)) and len(f0) >= 2:
        # The legacy wizard tuple shape (freq, amp[, idx]).
        out["f0"] = peak_to_dict(PeakInfo(
            frequency=float(f0[0]), amplitude=float(f0[1]),
            index=int(f0[2]) if len(f0) > 2 else 0,
            label="f0", source="manual"))
    else:
        out["f0"] = None
    sec: List[Dict[str, Any]] = []
    for i, s in enumerate(pdata.get("secondary") or []):
        if isinstance(s, PeakInfo):
            sec.append(peak_to_dict(s))
        elif isinstance(s, dict):
            sec.append(peak_to_dict(peak_from_dict(s)))
        elif isinstance(s, (tuple, list)) and len(s) >= 2:
            sec.append(peak_to_dict(PeakInfo(
                frequency=float(s[0]), amplitude=float(s[1]),
                index=int(s[2]) if len(s) > 2 else 0,
                label=f"sec{i + 1}", source="manual")))
    out["secondary"] = sec
    for key in ("vs30", "vsavg", "bedrock_depth"):
        out[key] = pdata.get(key)
    return out


# ---------------------------------------------------------------------------
# The legacy-identical CSV / vs_results writers
# ---------------------------------------------------------------------------


def write_summary_peak(csv_path: Path, peak_freq: float,
                       peak_amp: float) -> bool:
    """Update ``Peak_Frequency_Hz`` / ``Peak_Amplitude`` in an existing
    summary CSV — byte-compatible port of the legacy
    ``_update_summary_csv`` (same ``%.6f`` formatting, same row-1 write)."""
    try:
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return False
        header = rows[0]
        freq_col = amp_col = None
        for i, h in enumerate(header):
            if h.strip() == "Peak_Frequency_Hz":
                freq_col = i
            elif h.strip() == "Peak_Amplitude":
                amp_col = i
        if freq_col is None or amp_col is None:
            return False
        rows[1][freq_col] = f"{peak_freq:.6f}"
        rows[1][amp_col] = f"{peak_amp:.6f}"
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        return True
    except Exception as exc:  # noqa: BLE001 — mirror legacy best-effort
        logger.warning("summary update failed for %s: %s", csv_path, exc)
        return False


def create_minimal_summary(csv_path: Path, step_name: str,
                           peak_freq: float, peak_amp: float) -> bool:
    """Create a minimal summary CSV the report generator can read — port of
    the legacy ``_create_minimal_summary_csv`` (identical header + parse)."""
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "N_Finite_Layers",
                             "Peak_Frequency_Hz", "Peak_Amplitude"])
            parts = step_name.split("_")
            step_num = parts[0].replace("Step", "")
            n_layers = parts[1].split("-")[0] if len(parts) > 1 else "?"
            writer.writerow([f"Step{step_num}", n_layers,
                             f"{peak_freq:.6f}", f"{peak_amp:.6f}"])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("minimal summary failed for %s: %s", csv_path, exc)
        return False


def _write_vs_results(strip_path: Path, vs_data: Dict[str, Any]) -> None:
    """Legacy-identical ``vs_results.json`` (indent=2, default=str)."""
    with open(strip_path / VS_RESULTS_NAME, "w") as f:
        json.dump(vs_data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------


def resolve_step_folder(strip_dir: Path, step: str) -> Optional[Path]:
    """Match *step* to a step folder: exact name first, then the
    ``Step<n>`` / ``Step<n>_...`` prefix."""
    strip_dir = Path(strip_dir)
    exact = strip_dir / step
    if exact.is_dir():
        return exact
    for folder in find_step_folders(strip_dir):
        if folder.name == step or folder.name.split("_")[0] == step:
            return folder
    return None


def persist_peaks(strip_dir: str,
                  picks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Write picked peaks + Vs context into a strip results tree.

    Parameters
    ----------
    strip_dir : str
        The directory containing the ``Step*_*-layer`` folders.
    picks : dict
        ``{step: {"f0": peak, "secondary": [peaks], "vs30": …,
        "vsavg": …, "bedrock_depth": …}}`` — *step* may be the folder
        name or ``Step<n>``; peaks may be :class:`PeakInfo`, dicts, or
        the legacy ``(freq, amp[, idx])`` tuples.

    Writes the ``picked_peaks.json`` sidecar (full fidelity, secondaries
    included), updates/creates the per-step summary CSVs (f0 only — the
    frozen reporter's contract), and ``vs_results.json``.
    """
    strip_path = Path(strip_dir)
    if not strip_path.is_dir():
        return {"success": False,
                "error": f"strip directory not found: {strip_dir}"}

    normalized: Dict[str, Dict[str, Any]] = {}
    updated: List[str] = []
    created: List[str] = []
    unmatched: List[str] = []
    vs_data: Dict[str, Any] = {}

    for step, pdata in picks.items():
        folder = resolve_step_folder(strip_path, str(step))
        if folder is None:
            unmatched.append(str(step))
            continue
        norm = _normalize_step_picks(pdata or {})
        normalized[folder.name] = norm

        f0 = norm.get("f0")
        if f0:
            summary_files = list(folder.glob("*summary*.csv"))
            if summary_files:
                for sf in summary_files:
                    if write_summary_peak(sf, f0["frequency"],
                                          f0["amplitude"]):
                        updated.append(str(sf))
            else:
                sf = folder / "step_summary.csv"
                if create_minimal_summary(sf, folder.name,
                                          f0["frequency"],
                                          f0["amplitude"]):
                    created.append(str(sf))

        if any(norm.get(k) for k in ("vs30", "vsavg", "bedrock_depth")):
            vs_data[folder.name] = {
                "vs30": norm.get("vs30"),
                "vsavg": norm.get("vsavg"),
                "bedrock_depth": norm.get("bedrock_depth"),
            }

    if vs_data:
        _write_vs_results(strip_path, vs_data)

    sidecar = strip_path / SIDECAR_NAME
    payload = {"version": SIDECAR_VERSION, "steps": normalized}
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)

    return {
        "success": True,
        "sidecar": str(sidecar),
        "updated_summaries": updated,
        "created_summaries": created,
        "unmatched_steps": unmatched,
        "vs_steps": sorted(vs_data),
    }


# ---------------------------------------------------------------------------
# Load picks (sidecar first, legacy fallback)
# ---------------------------------------------------------------------------


def _read_summary_peak(folder: Path) -> Optional[Tuple[float, float]]:
    """Read (Peak_Frequency_Hz, Peak_Amplitude) from a step's summary."""
    for sf in sorted(folder.glob("*summary*.csv")):
        try:
            with open(sf, "r", newline="") as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                continue
            header = [h.strip() for h in rows[0]]
            fi = header.index("Peak_Frequency_Hz")
            ai = header.index("Peak_Amplitude")
            return float(rows[1][fi]), float(rows[1][ai])
        except (ValueError, IndexError):
            continue
    return None


def load_picks(strip_dir: str) -> Dict[str, Any]:
    """Load picks from a results tree.

    Returns ``{"steps": {folder_name: step-picks}, "source": "sidecar" |
    "legacy" | None}``.  The legacy fallback reads ``vs_results.json``
    (Vs context) and marks every step's summary peak as ``source="auto"``
    (legacy trees never persisted manual secondaries — nothing to
    recover there).
    """
    strip_path = Path(strip_dir)
    sidecar = strip_path / SIDECAR_NAME
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            return {"steps": payload.get("steps", {}), "source": "sidecar"}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("unreadable sidecar %s: %s", sidecar, exc)

    steps: Dict[str, Any] = {}
    vs_path = strip_path / VS_RESULTS_NAME
    vs_data: Dict[str, Any] = {}
    if vs_path.is_file():
        try:
            vs_data = json.loads(vs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            vs_data = {}
    for folder in find_step_folders(strip_path):
        entry: Dict[str, Any] = {"f0": None, "secondary": [],
                                 "vs30": None, "vsavg": None,
                                 "bedrock_depth": None}
        peak = _read_summary_peak(folder)
        if peak is not None:
            entry["f0"] = peak_to_dict(PeakInfo(
                frequency=peak[0], amplitude=peak[1],
                label="f0", source="auto"))
        for key, val in (vs_data.get(folder.name) or {}).items():
            if key in entry:
                entry[key] = val
        steps[folder.name] = entry
    return {"steps": steps, "source": "legacy" if steps else None}


# ---------------------------------------------------------------------------
# Rehydrate a results folder (no recompute)
# ---------------------------------------------------------------------------


def _read_hv_curve(folder: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``hv_curve.csv`` (header ``Frequency_Hz,HVSR_Amplitude``)."""
    path = folder / "hv_curve.csv"
    freqs: List[float] = []
    amps: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and header[0].strip().lower() not in (
                "frequency_hz", "frequency"):
            # headerless variant — first row is data
            try:
                freqs.append(float(header[0]))
                amps.append(float(header[1]))
            except (ValueError, IndexError):
                pass
        for row in reader:
            try:
                freqs.append(float(row[0]))
                amps.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return np.asarray(freqs, dtype=float), np.asarray(amps, dtype=float)


def _parse_step_folder_name(name: str) -> Tuple[int, int]:
    """``Step3_4-layer`` → (3, 4); best-effort."""
    step_num = 0
    n_layers = 0
    parts = name.split("_")
    try:
        step_num = int(parts[0].replace("Step", ""))
    except (ValueError, IndexError):
        pass
    if len(parts) > 1:
        try:
            n_layers = int(parts[1].split("-")[0])
        except (ValueError, IndexError):
            pass
    return step_num, n_layers


def rehydrate_results_folder(strip_dir: str) -> Dict[str, Any]:
    """Rebuild a :class:`StripResult` (+ picks) from a results tree.

    NO recompute: curves come from each step's ``hv_curve.csv``; peaks
    from the picks (sidecar or legacy summaries).  Works on trees the
    legacy app produced.

    Returns ``{"success", "result": StripResult, "picks": dict,
    "picks_source": str | None}`` (the caller serialises; the live
    ``StripResult`` object is for session storage).
    """
    strip_path = Path(strip_dir)
    if not strip_path.is_dir():
        return {"success": False,
                "error": f"not a directory: {strip_dir}"}
    step_folders = find_step_folders(strip_path)
    if not step_folders:
        return {"success": False,
                "error": f"no Step*_*-layer folders in: {strip_dir}"}

    picks_env = load_picks(str(strip_path))
    picks = picks_env["steps"]

    steps: List[StepResult] = []
    for folder in step_folders:
        step_num, n_layers = _parse_step_folder_name(folder.name)
        try:
            freqs, amps = _read_hv_curve(folder)
        except (OSError, StopIteration):
            steps.append(StepResult(
                step_number=step_num, n_layers=n_layers,
                success=False, error="hv_curve.csv missing/unreadable"))
            continue
        models = sorted(folder.glob("model_*.txt"))
        model_path = str(models[0]) if models else ""

        f0 = (picks.get(folder.name) or {}).get("f0")
        if f0:
            pf, pa = float(f0["frequency"]), float(f0["amplitude"])
        elif len(amps):
            idx = int(np.argmax(amps))
            pf, pa = float(freqs[idx]), float(amps[idx])
        else:
            pf = pa = 0.0

        steps.append(StepResult(
            step_number=step_num,
            n_layers=n_layers,
            model_path=model_path,
            frequencies=freqs,
            amplitudes=amps,
            peak_frequency=pf,
            peak_amplitude=pa,
            success=True,
        ))

    result = StripResult(
        initial_profile=strip_path.name,
        output_directory=str(strip_path.parent),
        strip_directory=str(strip_path),
        steps=steps,
        success=True,
    )
    return {
        "success": True,
        "result": result,
        "picks": picks,
        "picks_source": picks_env["source"],
    }


__all__ = [
    "SIDECAR_NAME",
    "SIDECAR_VERSION",
    "VS_RESULTS_NAME",
    "peak_to_dict",
    "peak_from_dict",
    "persist_peaks",
    "load_picks",
    "rehydrate_results_folder",
    "resolve_step_folder",
    "write_summary_peak",
    "create_minimal_summary",
]
