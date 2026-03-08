import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Reuse your existing forward modeling module and its settings/functions
import HV_In_Forward_Modeling as hvmod


def _rows_from_model_lines(model_lines: List[str]) -> List[List[float]]:
    """Convert model_lines (first element is N) into list of [thk, vp, vs, rho] floats."""
    if not model_lines:
        raise ValueError("Empty model_lines")
    try:
        n = int(model_lines[0])
    except Exception as exc:
        raise ValueError("First line of model_lines must be integer N") from exc
    rows = []
    for line in model_lines[1:1 + n]:
        parts = [float(x) for x in line.split()[:4]]
        if len(parts) < 4:
            raise ValueError("Model line has fewer than 4 numeric columns")
        rows.append(parts)
    if not rows or rows[-1][0] != 0.0:
        raise ValueError("Last model row must be half-space (thickness=0)")
    return rows


def _model_lines_from_rows(rows: List[List[float]]) -> List[str]:
    n = len(rows)
    lines = [str(n)]
    for r in rows:
        lines.append(" ".join(f"{x:.12g}" for x in r))
    return lines


def _load_base_model_lines() -> List[str]:
    """Use the same inputs as hvmod: prefer model_prefix triplet; otherwise models_file."""
    if getattr(hvmod, "model_prefix", ""):
        ml = hvmod.parse_triplet_from_prefix(hvmod.model_prefix)
        if ml is None:
            # Fallback to single VS file if prefix could not be used
            return hvmod.parse_vs_any(hvmod.models_file)
        return ml
    return hvmod.parse_vs_any(hvmod.models_file)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_hv_csv(csv_path: Path, freqs: np.ndarray, amps: np.ndarray) -> None:
    _ensure_dir(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Frequency[Hz]", "H/V_Amplitude"])
        for fval, aval in zip(freqs, amps):
            w.writerow([f"{float(fval):.6f}", f"{float(aval):.6f}"])


def _save_model_txt(txt_path: Path, model_lines: List[str]) -> None:
    _ensure_dir(txt_path)
    txt_path.write_text("\n".join(model_lines) + "\n", encoding="utf-8")


def _format_vs_profile(rows: List[List[float]]) -> str:
    # Represent profile as "thk vs; thk vs; ..." including half-space
    pairs = [f"{r[0]:.12g} {r[2]:.12g}" for r in rows]
    return "; ".join(pairs)


def _peak(freqs: np.ndarray, amps: np.ndarray) -> Tuple[float, float, int]:
    idx = int(np.argmax(amps))
    return float(freqs[idx]), float(amps[idx]), idx


def main():
    # Derive output directory based on the existing output_file
    base_out_dir = Path(hvmod.output_file).parent if getattr(hvmod, "output_file", None) else Path.cwd()
    out_dir = base_out_dir / "peel_layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    # To avoid many pop-up windows, you can disable interactive showing here if desired
    # hvmod.show_plot = False

    # Load and normalize base model
    base_model_lines = _load_base_model_lines()
    rows = _rows_from_model_lines(base_model_lines)

    # Partition into finite layers and half-space
    if not rows or rows[-1][0] != 0.0:
        raise RuntimeError("Model must end with a half-space row (thickness = 0)")
    finite_layers = [r for r in rows[:-1] if r[0] > 0]
    orig_hs = rows[-1]

    # Prepare summary CSV
    summary_rows = []
    summary_csv = out_dir / "peaks_summary.csv"

    # Build step models
    total_finite = len(finite_layers)
    if total_finite == 0:
        # Only half-space
        model_lines = _model_lines_from_rows([orig_hs])
        hv_text = hvmod.run_hvf_for_model(model_lines)
        freqs, amps = hvmod.parse_hv_text(hv_text)
        fpk, Apk, _ = _peak(freqs, amps)
        _save_hv_csv(out_dir / "hv_step_0.csv", freqs, amps)
        hv_png = out_dir / "hv_step_0.png"
        hvmod.plot_hv_with_peak(freqs, amps, str(hv_png), title="H/V (step 0)")
        _save_model_txt(out_dir / "model_step_0.txt", model_lines)
        summary_rows.append({
            "step_index": 0,
            "layers_remaining": 0,
            "promoted_to_halfspace_vs": "",
            "peak_freq_hz": f"{fpk:.6f}",
            "peak_amp": f"{Apk:.6f}",
            "model_rows": _format_vs_profile([orig_hs]),
        })
    else:
        # Step 0: full model (finite + original HS)
        step_idx = 0
        model_rows = finite_layers + [orig_hs]
        model_lines = _model_lines_from_rows(model_rows)
        hv_text = hvmod.run_hvf_for_model(model_lines)
        freqs, amps = hvmod.parse_hv_text(hv_text)
        fpk, Apk, _ = _peak(freqs, amps)
        _save_hv_csv(out_dir / f"hv_step_{step_idx}.csv", freqs, amps)
        hv_png = out_dir / f"hv_step_{step_idx}.png"
        hvmod.plot_hv_with_peak(freqs, amps, str(hv_png), title=f"H/V (step {step_idx})")
        _save_model_txt(out_dir / f"model_step_{step_idx}.txt", model_lines)
        summary_rows.append({
            "step_index": step_idx,
            "layers_remaining": len(finite_layers),
            "promoted_to_halfspace_vs": "",
            "peak_freq_hz": f"{fpk:.6f}",
            "peak_amp": f"{Apk:.6f}",
            "model_rows": _format_vs_profile(model_rows),
        })

        # Peel from the bottom until only one finite + HS remain
        # For s in 1..(total_finite-1)
        for s in range(1, total_finite):
            step_idx = s
            top_count = total_finite - s
            top_layers = finite_layers[:top_count]
            promoted = finite_layers[top_count]  # layer to promote to HS
            hs_now = [0.0, promoted[1], promoted[2], promoted[3]]
            model_rows = top_layers + [hs_now]
            model_lines = _model_lines_from_rows(model_rows)

            hv_text = hvmod.run_hvf_for_model(model_lines)
            freqs, amps = hvmod.parse_hv_text(hv_text)
            fpk, Apk, _ = _peak(freqs, amps)

            # Save artifacts
            _save_hv_csv(out_dir / f"hv_step_{step_idx}.csv", freqs, amps)
            hv_png = out_dir / f"hv_step_{step_idx}.png"
            hvmod.plot_hv_with_peak(freqs, amps, str(hv_png), title=f"H/V (step {step_idx})")
            _save_model_txt(out_dir / f"model_step_{step_idx}.txt", model_lines)

            summary_rows.append({
                "step_index": step_idx,
                "layers_remaining": len(top_layers),
                "promoted_to_halfspace_vs": f"{promoted[2]:.12g}",
                "peak_freq_hz": f"{fpk:.6f}",
                "peak_amp": f"{Apk:.6f}",
                "model_rows": _format_vs_profile(model_rows),
            })

    # Write summary CSV
    _ensure_dir(summary_csv)
    fieldnames = [
        "step_index",
        "layers_remaining",
        "promoted_to_halfspace_vs",
        "peak_freq_hz",
        "peak_amp",
        "model_rows",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"[INFO] Peel-layers outputs written under: {out_dir}")


if __name__ == "__main__":
    main()


