"""Generate the SC-1 legacy picked-run fixture (spec 002, task T01).

Run ONCE from the distribution root (its own process — it imports the
LEGACY PyQt5 module to drive the real legacy persistence code paths):

    python tests_v2/fixtures/legacy_picked_run/_generate.py

Steps:
1. Run a deterministic sh_wave strip of ``examples/different_files/
   example_model.txt`` into ``run/`` (report off).
2. Apply a fixed manual-picking walkthrough via the LEGACY writers —
   ``HVStripWindow._update_summary_csv`` / ``_create_minimal_summary_csv``
   (real static methods) and the ``_persist_vs_data`` json shape — i.e.
   exactly what the legacy wizard's Finish persisted.  NOTE the legacy
   truth this captures: only f0 reaches the CSVs; the picked SECONDARIES
   are dropped (spec 002 fixes that — the new api's sidecar keeps them).
3. Write ``picks.json`` (the walkthrough, sidecar-shaped) so tests replay
   the SAME picks through the new api and compare outputs.
4. Strip figure binaries (*.png/*.pdf) — the fixture pins values, not
   pixels.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent          # the distribution root
sys.path.insert(0, str(ROOT))

RUN_DIR = HERE / "run"
MODEL = ROOT / "examples" / "different_files" / "example_model.txt"

#: The fixed walkthrough (off-grid frequencies = exact-click semantics;
#: amplitudes as the legacy wizard would interpolate them).  Keyed by step
#: NUMBER; resolved to folder names after the run.
PICKS = {
    0: {"f0": (4.10, 3.30), "secondary": [(8.50, 1.90), (12.30, 1.50)],
        "vs30": 252.7, "vsavg": 341.2, "bedrock_depth": 44.0},
    1: {"f0": (4.60, 3.10), "secondary": [(9.10, 1.70)],
        "vs30": 261.4, "vsavg": 355.9, "bedrock_depth": 38.0},
    2: {"f0": (5.25, 2.85), "secondary": [],
        "vs30": 270.0, "vsavg": None, "bedrock_depth": None},
}


def main() -> None:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)

    # ── 1. the deterministic sh_wave strip ─────────────────────────────
    from HV_Strip_Progressive.api import HVStripAnalysis

    a = HVStripAnalysis()
    a.set_engine(name="sh_wave")
    a.set_strip(generate_report=False)
    a.load_profile_from_file(str(MODEL), name="example_model")
    env = a.run_stripping_for_profile("example_model",
                                      output_dir=str(RUN_DIR))
    assert env.get("success"), env
    strip_dir = Path(env["strip_directory"])
    print(f"strip tree: {strip_dir}")

    from HV_Strip_Progressive.core.batch_workflow import find_step_folders

    folders = find_step_folders(strip_dir)
    assert len(folders) >= 3, folders

    # ── 2. the LEGACY persistence, on the real legacy code ─────────────
    from HV_Strip_Progressive.gui.strip_window import HVStripWindow

    vs_data = {}
    picks_by_folder = {}
    for num, pdata in PICKS.items():
        folder = folders[num]
        picks_by_folder[folder.name] = pdata
        f0 = pdata["f0"]
        summaries = list(folder.glob("*summary*.csv"))
        if summaries:
            for sf in summaries:
                HVStripWindow._update_summary_csv(sf, f0[0], f0[1])
        else:
            HVStripWindow._create_minimal_summary_csv(
                folder / "step_summary.csv", folder.name, f0[0], f0[1])
        if pdata.get("vs30") or pdata.get("vsavg") or \
                pdata.get("bedrock_depth"):
            vs_data[folder.name] = {
                "vs30": pdata.get("vs30"),
                "vsavg": pdata.get("vsavg"),
                "bedrock_depth": pdata.get("bedrock_depth"),
            }
    # _persist_vs_data verbatim (instance method; body replicated exactly).
    with open(strip_dir / "vs_results.json", "w") as f:
        json.dump(vs_data, f, indent=2, default=str)

    # ── 3. the replayable picks (sidecar-shaped) ───────────────────────
    payload = {
        "strip_dir_rel": str(strip_dir.relative_to(HERE)).replace("\\", "/"),
        "steps": {
            name: {
                "f0": {"frequency": p["f0"][0], "amplitude": p["f0"][1],
                       "label": "f0", "source": "manual"},
                "secondary": [
                    {"frequency": s[0], "amplitude": s[1],
                     "label": f"sec{i + 1}", "source": "manual"}
                    for i, s in enumerate(p["secondary"])],
                "vs30": p.get("vs30"),
                "vsavg": p.get("vsavg"),
                "bedrock_depth": p.get("bedrock_depth"),
            }
            for name, p in picks_by_folder.items()
        },
    }
    (HERE / "picks.json").write_text(json.dumps(payload, indent=2),
                                     encoding="utf-8")

    # ── 4. drop binaries ───────────────────────────────────────────────
    removed = 0
    for pattern in ("*.png", "*.pdf"):
        for f in RUN_DIR.rglob(pattern):
            f.unlink()
            removed += 1
    print(f"removed {removed} figure binaries")
    print("fixture ready")


if __name__ == "__main__":
    main()
