"""
HV forward computation module.

Responsibilities:
- Run HVf.exe on a single HVf-format model file and return frequencies/amplitudes.
- Provide robust parsing of HV.dat (multi-line or single-line token pairs).
- Simple configuration dict to set exe path and frequency grid.

Usage (as a module):
  from hv_forward import compute_hv_curve
  freqs, amps = compute_hv_curve(model_path, config)
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List


DEFAULT_CONFIG = {
    "exe_path": str(Path("HVf.exe").resolve()),
    "fmin": 0.2,
    "fmax": 20.0,
    "nf": 71,
    "nmr": 10,
    "nml": 10,
    "nks": 10,
}


def _strip_comment(line: str) -> str:
    return line.split('#', 1)[0].strip()


def _parse_hv_text(hv_text: str) -> Tuple[List[float], List[float]]:
    """Parse HV.dat text into frequency and amplitude arrays.

    Supports two common formats:
    - Two-column text with optional comments
    - Single-line sequence of tokens f1 a1 f2 a2 ...
    """
    rows: List[Tuple[float, float]] = []
    for ln in hv_text.splitlines():
        core = _strip_comment(ln)
        if not core:
            continue
        parts = core.split()
        if len(parts) >= 2:
            try:
                f = float(parts[0])
                a = float(parts[1])
                rows.append((f, a))
            except Exception:
                continue

    # Fallback: single-line tokens
    if len(rows) < 4:
        toks: List[float] = []
        for tok in hv_text.split():
            try:
                toks.append(float(tok))
            except Exception:
                pass
        if len(toks) >= 4 and len(toks) % 2 == 0:
            rows = [(toks[i], toks[i + 1]) for i in range(0, len(toks), 2)]

    if not rows:
        raise ValueError("Could not parse HV output")

    freqs = [r[0] for r in rows]
    amps = [r[1] for r in rows]
    # Heuristic: if amps look increasing monotonic and freqs don't, swap
    def _mostly_increasing(arr: List[float]) -> bool:
        if len(arr) < 3:
            return False
        diffs = [arr[i+1] - arr[i] for i in range(len(arr)-1)]
        nonneg = sum(1 for d in diffs if d >= -1e-12)
        return nonneg >= 0.8 * len(diffs)

    if _mostly_increasing(amps) and not _mostly_increasing(freqs):
        freqs, amps = amps, freqs

    # Basic validation
    if any(f < 0 for f in freqs):
        raise ValueError("Negative frequencies in HV output")
    if any(a < 0 for a in amps):
        raise ValueError("Negative amplitudes in HV output")

    return freqs, amps


def compute_hv_curve(model_path: str, config: Dict = None) -> Tuple[List[float], List[float]]:
    """Run HVf.exe for the provided HVf-format model file and return (freqs, amps)."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    exe_path = Path(cfg["exe_path"])  # allow absolute or relative
    if not exe_path.exists():
        raise FileNotFoundError(f"HVf.exe not found at: {exe_path}")

    # Create temporary directory to write model and capture HV.dat
    with tempfile.TemporaryDirectory() as tdir:
        tdir_p = Path(tdir)
        model_file = tdir_p / "model.txt"

        # Copy or write the model content into temp for safety
        original = Path(model_path)
        text = original.read_text(encoding='utf-8', errors='ignore')
        model_file.write_text(text, encoding='utf-8')

        cmd = [
            str(exe_path),
            "-hv",
            "-f", str(model_file),
            "-fmin", str(cfg["fmin"]),
            "-fmax", str(cfg["fmax"]),
            "-nf", str(cfg["nf"]),
            "-nmr", str(cfg["nmr"]),
            "-nml", str(cfg["nml"]),
            "-nks", str(cfg["nks"]),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=tdir_p,
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"HVf.exe failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("HVf.exe timed out")

        hv_path = tdir_p / "HV.dat"
        if hv_path.exists():
            hv_text = hv_path.read_text(encoding='utf-8', errors='ignore')
        elif result.stdout.strip():
            hv_text = result.stdout
        else:
            raise RuntimeError("No HV output generated (HV.dat missing and empty stdout)")

    return _parse_hv_text(hv_text)


__all__ = [
    "compute_hv_curve",
    "DEFAULT_CONFIG",
]
