"""The GOLDEN COMPUTE LOCK — the frozen-engine proof for the house-style rebuild.

Runs the canonical example model through ``run_complete_workflow`` and pins the
per-step HV-curve numerics against committed fixtures.  Two tiers:

* **sh_wave** — the pure-Python engine; runs on any machine (the default gate).
* **diffuse_field** — the vendored ``HVf.exe`` subprocess; ``@pytest.mark.hvf``
  (opt-in: ``pytest -m hvf``), only where the exe runs.

REGENERATING (only when a compute change is INTENDED):
    python tests/test_golden_compute.py --regen
Any unintended fixture diff = a compute regression; the rebuild must keep these
bit-identical (constraint: performance/numerics frozen).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXAMPLE_MODEL = PROJECT_ROOT / "examples" / "different_files" / "example_model.txt"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _run(engine_name: str, out_dir: Path) -> dict:
    from HV_Strip_Progressive.core.batch_workflow import run_complete_workflow

    out_dir.mkdir(parents=True, exist_ok=True)
    return run_complete_workflow(
        str(EXAMPLE_MODEL), str(out_dir),
        workflow_config={"engine_name": engine_name},
        engine_name=engine_name,
    )


def _digest(result: dict) -> dict:
    """A JSON-stable numeric digest of a workflow result.

    Per step: the recorded scalars (peak frequency/amplitude/index, Vs30,
    sample count) PLUS a checksum of the full HV curve read back from the
    step's ``hv_curve.csv`` — enough to detect ANY numeric drift without
    committing megabytes.  Must run while the output directory still exists.
    """
    steps = {}
    for name, step in sorted((result.get("step_results") or {}).items()):
        entry = {}
        for key in ("n_frequencies", "peak_index"):
            if step.get(key) is not None:
                entry[key] = int(step[key])
        for key in ("peak_frequency", "peak_amplitude", "vs30"):
            if step.get(key) is not None:
                entry[key] = round(float(step[key]), 9)
        if step.get("vs30_extrapolated") is not None:
            entry["vs30_extrapolated"] = bool(step["vs30_extrapolated"])
        csv_path = step.get("hv_csv")
        if csv_path and Path(csv_path).is_file():
            data = np.genfromtxt(str(csv_path), delimiter=",", skip_header=1)
            freqs, amps = data[:, 0], data[:, 1]
            entry.update({
                "curve_n": int(freqs.size),
                "f_first": round(float(freqs[0]), 9),
                "f_last": round(float(freqs[-1]), 9),
                "amp_sum": round(float(np.sum(amps)), 6),
                "amp_max": round(float(np.max(amps)), 9),
                "f_at_max": round(float(freqs[int(np.argmax(amps))]), 9),
            })
        steps[name] = entry
    return {"success": bool(result.get("success")), "steps": steps}


def _golden_path(engine_name: str) -> Path:
    return GOLDEN_DIR / f"workflow_{engine_name}.json"


def _check_or_fail(engine_name: str, tmp_path: Path) -> None:
    golden_file = _golden_path(engine_name)
    if not golden_file.is_file():
        pytest.skip(
            f"golden fixture missing — regenerate with "
            f"`python {Path(__file__).name} --regen`"
        )
    result = _run(engine_name, tmp_path / f"golden_{engine_name}")
    got = _digest(result)
    want = json.loads(golden_file.read_text(encoding="utf-8"))
    assert got == want, (
        f"COMPUTE DRIFT for engine '{engine_name}' — the frozen workflow "
        f"produced different numbers than the committed golden fixture."
    )


def test_workflow_golden_sh_wave(tmp_path):
    _check_or_fail("sh_wave", tmp_path)


@pytest.mark.hvf
def test_workflow_golden_diffuse_field(tmp_path):
    exe = (PROJECT_ROOT / "HV_Strip_Progressive" / "core" / "engines"
           / "diffuse_wave_field" / "exe_Win" / "HVf.exe")
    if not exe.is_file():
        pytest.skip("HVf.exe not present")
    _check_or_fail("diffuse_field", tmp_path)


# ----------------------------------------------------------------------
#  Regeneration entry point (NOT a test): python test_golden_compute.py --regen
# ----------------------------------------------------------------------
def _regen() -> None:
    import tempfile

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for engine in ("sh_wave", "diffuse_field"):
        try:
            with tempfile.TemporaryDirectory() as td:
                digest = _digest(_run(engine, Path(td) / "out"))
        except Exception as exc:                              # noqa: BLE001
            print(f"[skip] {engine}: {exc}")
            continue
        path = _golden_path(engine)
        path.write_text(json.dumps(digest, indent=1, sort_keys=True),
                        encoding="utf-8")
        print(f"[ok] wrote {path} ({len(digest['steps'])} steps)")


if __name__ == "__main__":
    if "--regen" in sys.argv:
        _regen()
    else:
        print(__doc__)
