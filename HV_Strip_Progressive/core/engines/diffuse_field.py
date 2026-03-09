"""
Diffuse Wavefield Theory engine — wraps HVf.exe.

This is the default forward-modeling engine.  It computes theoretical
H/V spectral ratios using the diffuse wavefield assumption implemented
by the external ``HVf`` executable.
"""

import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .base import BaseForwardEngine, EngineResult


def _get_default_exe_path() -> str:
    """Auto-detect the HVf executable based on OS.

    Looks in the ``diffuse_wave_field`` directory that sits alongside this
    module: ``core/engines/diffuse_wave_field/exe_Win/`` (Windows) or
    ``core/engines/diffuse_wave_field/exe_Linux/`` (Linux).
    """
    dwf_dir = Path(__file__).parent / "diffuse_wave_field"

    system = platform.system()
    if system == "Windows":
        candidates = [dwf_dir / "exe_Win" / "HVf.exe"]
    elif system == "Linux":
        candidates = [
            dwf_dir / "exe_Linux" / "HVf",
            dwf_dir / "exe_Linux" / "HVf_Serial",
        ]
    else:
        candidates = [
            dwf_dir / "exe_Linux" / "HVf",
            dwf_dir / "exe_Linux" / "HVf_Serial",
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return "HVf.exe"


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_hv_text(hv_text: str) -> Tuple[List[float], List[float]]:
    """Parse HV.dat output into (freqs, amps).

    Supports two-column text and single-line token-pair formats.
    """
    rows: List[Tuple[float, float]] = []
    for ln in hv_text.splitlines():
        core = _strip_comment(ln)
        if not core:
            continue
        parts = core.split()
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue

    # Fallback: single-line tokens
    if len(rows) < 4:
        toks: List[float] = []
        for tok in hv_text.split():
            try:
                toks.append(float(tok))
            except ValueError:
                pass
        if len(toks) >= 4 and len(toks) % 2 == 0:
            rows = [(toks[i], toks[i + 1]) for i in range(0, len(toks), 2)]

    if not rows:
        raise ValueError("Could not parse HV output")

    freqs = [r[0] for r in rows]
    amps = [r[1] for r in rows]

    def _mostly_increasing(arr: List[float]) -> bool:
        if len(arr) < 3:
            return False
        diffs = [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
        return sum(1 for d in diffs if d >= -1e-12) >= 0.8 * len(diffs)

    if _mostly_increasing(amps) and not _mostly_increasing(freqs):
        freqs, amps = amps, freqs

    if any(f < 0 for f in freqs):
        raise ValueError("Negative frequencies in HV output")
    if any(a < 0 for a in amps):
        raise ValueError("Negative amplitudes in HV output")

    return freqs, amps


class DiffuseFieldEngine(BaseForwardEngine):
    """HVf.exe — diffuse wavefield H/V computation."""

    name = "diffuse_field"
    description = (
        "Diffuse wavefield theory (HVf.exe). Computes H/V spectral "
        "ratios assuming a diffuse wavefield in a horizontally layered "
        "medium."
    )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_default_config(self) -> Dict:
        return {
            "exe_path": _get_default_exe_path(),
            "fmin": 0.2,
            "fmax": 20.0,
            "nf": 71,
            "nmr": 10,
            "nml": 10,
            "nks": 10,
        }

    def get_required_params(self) -> List[str]:
        return ["exe_path", "fmin", "fmax", "nf"]

    def validate_config(self, config: Dict) -> Tuple[bool, str]:
        cfg = {**self.get_default_config(), **(config or {})}
        exe = Path(cfg["exe_path"])
        if not exe.exists():
            return False, f"HVf executable not found: {exe}"
        if cfg["fmin"] >= cfg["fmax"]:
            return False, "fmin must be less than fmax"
        if cfg["nf"] < 2:
            return False, "nf must be >= 2"
        return True, ""

    # ------------------------------------------------------------------
    # Model formatting
    # ------------------------------------------------------------------

    def format_model(self, profile) -> str:
        """Convert ``SoilProfile`` to HVf format (thickness vp vs density)."""
        return profile.to_hvf_format()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self, model_path: str, config: Dict = None) -> EngineResult:
        cfg = {**self.get_default_config(), **(config or {})}
        exe_path = Path(cfg["exe_path"])
        if not exe_path.exists():
            raise FileNotFoundError(f"HVf.exe not found at: {exe_path}")

        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory() as tdir:
            tdir_p = Path(tdir)
            model_file = tdir_p / "model.txt"
            text = Path(model_path).read_text(encoding="utf-8", errors="ignore")
            model_file.write_text(text, encoding="utf-8")

            # Copy exe into temp dir to avoid long-path / permission issues
            local_exe = tdir_p / exe_path.name
            try:
                shutil.copy2(str(exe_path), str(local_exe))
            except (PermissionError, OSError):
                local_exe = exe_path  # fall back to original path

            cmd = [
                str(local_exe),
                "-hv",
                "-f", str(model_file),
                "-fmin", str(cfg["fmin"]),
                "-fmax", str(cfg["fmax"]),
                "-nf", str(cfg["nf"]),
                "-nmr", str(cfg["nmr"]),
                "-nml", str(cfg["nml"]),
                "-nks", str(cfg["nks"]),
            ]

            # Hide console window on Windows
            kwargs: Dict = {}
            if sys.platform == "win32":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 0  # SW_HIDE
                kwargs["startupinfo"] = si
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            try:
                result = subprocess.run(
                    cmd,
                    cwd=tdir_p,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120,
                    **kwargs,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"HVf.exe failed (exit code {e.returncode}):\n"
                    f"  stderr: {e.stderr}\n"
                    f"  stdout: {e.stdout}\n"
                    f"  exe: {local_exe}\n"
                    f"  Tip: check antivirus or run as administrator."
                ) from e
            except PermissionError as e:
                raise RuntimeError(
                    f"Permission denied running HVf.exe at {local_exe}.\n"
                    f"  Try: (1) whitelist in antivirus, (2) run as admin, "
                    f"(3) copy exe manually to a writable folder."
                ) from e
            except subprocess.TimeoutExpired as e:
                raise RuntimeError("HVf.exe timed out (120 s)") from e

            hv_path = tdir_p / "HV.dat"
            if hv_path.exists():
                hv_text = hv_path.read_text(encoding="utf-8", errors="ignore")
            elif result.stdout.strip():
                hv_text = result.stdout
            else:
                raise RuntimeError(
                    "No HV output generated (HV.dat missing and empty stdout)"
                )

        freqs, amps = _parse_hv_text(hv_text)
        elapsed = time.perf_counter() - t0

        return EngineResult(
            frequencies=np.array(freqs),
            amplitudes=np.array(amps),
            metadata={
                "engine": self.name,
                "exe_path": str(exe_path),
                "elapsed_seconds": elapsed,
            },
        )


__all__ = ["DiffuseFieldEngine"]
