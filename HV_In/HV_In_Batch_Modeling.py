#!/usr/bin/env python3
"""Batch H/V forward modeling using triplet Vs/Vp/rho model files.

This script scans the three text files that contain multiple layered models in
step format (as exported by Dinver), reconstructs each layered model, runs
HVf.exe for every model, and aggregates the resulting H/V curves into a single
output file tagged by model number.

Usage example:
    python HV_In_Batch_Modeling.py         --vs-file D:/Research/Analysis_tools/HV_In/example/Indep_Powerplant_B6_vs.txt         --vp-file D:/Research/Analysis_tools/HV_In/example/Indep_Powerplant_B6_vp.txt         --rho-file D:/Research/Analysis_tools/HV_In/example/Indep_Powerplant_B6_rho.txt         --count 5         --output D:/runs/batch_hv_curves.txt
"""
from __future__ import annotations

import argparse
import os
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HVFConfig:
    exe_path: Path
    fmin: float
    fmax: float
    nf: int
    nmr: int
    nml: int
    nks: int
    vp_vs_ratio_layers: float
    vp_vs_ratio_halfspace: float
    rho_mode: str
    rho_layers_value: float
    rho_halfspace_value: float
    rho_base: float
    rho_k: float


@dataclass(slots=True)
class ModelBlock:
    model_id: int
    pairs: list[tuple[float, float]]
    value: Optional[float]
    order: int
    source: Path


@dataclass(slots=True)
class HVCurve:
    model_id: int
    frequency: np.ndarray
    amplitude: np.ndarray
    value: Optional[float]


def _coerce_path(path_like: str | Path) -> Path:
    """Resolve input paths, translating Windows-style locations under WSL."""
    path_str = str(path_like)
    if os.name == "posix" and len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(path_like)


# ---------------------------------------------------------------------------
# Utilities shared with the single-model script (simplified copies)
# ---------------------------------------------------------------------------


def _float_or_none(token: str) -> Optional[float]:
    try:
        if token.lower() == "inf":
            return math.inf
        return float(token)
    except Exception:
        return None


def _same(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(1.0, abs(a), abs(b)))


def _vp_from_vs(cfg: HVFConfig, vs: float, *, is_halfspace: bool = False) -> float:
    ratio = cfg.vp_vs_ratio_halfspace if is_halfspace else cfg.vp_vs_ratio_layers
    vp = ratio * vs
    return max(vp, vs * 1.01)  # enforce Vp > Vs


def _rho_from_vs(cfg: HVFConfig, vs: float, *, is_halfspace: bool = False) -> float:
    if cfg.rho_mode == "constant":
        return cfg.rho_halfspace_value if is_halfspace else cfg.rho_layers_value
    return max(1000.0, cfg.rho_base + cfg.rho_k * vs)


# ---------------------------------------------------------------------------
# Parsing of step-formatted model files
# ---------------------------------------------------------------------------

_MODEL_RE = re.compile(r"#\s*Layered model\s+(\d+)(?::\s*value\s*=\s*([0-9eE.+-]+))?", re.IGNORECASE)


def _read_model_blocks(path: Path, label: str) -> list[ModelBlock]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks: list[ModelBlock] = []
    current_id: Optional[int] = None
    current_val: Optional[float] = None
    current_pairs: list[tuple[float, float]] = []
    order = 0

    for line in text:
        match = _MODEL_RE.match(line.strip())
        if match:
            if current_id is not None and current_pairs:
                if current_pairs[-1][1] is not math.inf:
                    current_pairs.append((current_pairs[-1][0], math.inf))
                blocks.append(ModelBlock(current_id, current_pairs, current_val, order, path))
                order += 1
            current_id = int(match.group(1))
            current_val = float(match.group(2)) if match.group(2) else None
            current_pairs = []
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        tokens = [_float_or_none(tok) for tok in stripped.split()]
        tokens = [t for t in tokens if t is not None]
        if len(tokens) >= 2:
            current_pairs.append((float(tokens[0]), float(tokens[1])))

    if current_id is not None and current_pairs:
        if current_pairs[-1][1] is not math.inf:
            current_pairs.append((current_pairs[-1][0], math.inf))
        blocks.append(ModelBlock(current_id, current_pairs, current_val, order, path))

    if not blocks:
        raise RuntimeError(f"No models found in {path} ({label}).")
    return blocks


def _segments_from_pairs(pairs: list[tuple[float, float]]):
    segments = []
    i = 0
    while i + 1 < len(pairs):
        v0, d0 = pairs[i]
        v1, d1 = pairs[i + 1]
        if (d0 not in (None, math.inf)) and (d1 not in (None, math.inf)) and _same(v0, v1) and d1 > d0:
            segments.append((float(d0), float(d1), float(v0)))
            i += 2
            continue
        if d0 == d1 and not _same(v0, v1):
            i += 1
            continue
        break

    if not segments:
        depth = 0.0
        for value, thickness in pairs:
            if thickness is math.inf:
                break
            thickness = float(thickness)
            if thickness > 1e-12:
                segments.append((depth, depth + thickness, float(value)))
                depth += thickness

    hs_value = None
    for k in range(len(pairs) - 1, -1, -1):
        value, depth = pairs[k]
        if depth is math.inf:
            hs_value = pairs[k - 1][0] if k > 0 else value
            break
    if hs_value is None:
        hs_value = segments[-1][2] if segments else pairs[-1][0]

    if segments and segments[0][0] > 1e-12:
        d0, d1, v = segments[0]
        segments[0] = (0.0, d1, v)

    return segments, float(hs_value)


def _value_at_depth(segments: list[tuple[float, float, float]], depth: float, default_value: float) -> float:
    if not segments:
        return default_value
    for start, end, value in segments:
        if depth >= start - 1e-12 and depth < end - 1e-12:
            return value
    if depth < segments[0][0]:
        return segments[0][2]
    return default_value


# ---------------------------------------------------------------------------
# Model assembly and HVf execution
# ---------------------------------------------------------------------------


def _build_model_lines(cfg: HVFConfig,
                        vs_pairs: list[tuple[float, float]],
                        vp_pairs: Optional[list[tuple[float, float]]],
                        rho_pairs: Optional[list[tuple[float, float]]]) -> list[str]:
    vs_segments, hs_vs = _segments_from_pairs(vs_pairs)
    if not vs_segments:
        raise RuntimeError("Vs pairs cannot be converted into segments.")

    vp_segments = None
    hs_vp = None
    if vp_pairs:
        vp_segments, hs_vp = _segments_from_pairs(vp_pairs)

    rho_segments = None
    hs_rho = None
    if rho_pairs:
        rho_segments, hs_rho = _segments_from_pairs(rho_pairs)

    breakpoints = {0.0}
    for segs in (vs_segments, vp_segments, rho_segments):
        if segs:
            for start, end, _ in segs:
                if start not in (None, math.inf):
                    breakpoints.add(float(start))
                if end not in (None, math.inf):
                    breakpoints.add(float(end))
    depths = sorted(bp for bp in breakpoints if bp >= 0)
    layers = []
    for idx in range(len(depths) - 1):
        d0, d1 = depths[idx], depths[idx + 1]
        thickness = d1 - d0
        if thickness <= 1e-12:
            continue
        mid = d0 + 0.5 * thickness
        vs_val = _value_at_depth(vs_segments, mid, hs_vs)
        vp_default = _vp_from_vs(cfg, vs_val, is_halfspace=False)
        vp_val = _value_at_depth(vp_segments, mid, hs_vp if hs_vp is not None else vp_default) if vp_segments else vp_default
        rho_default = _rho_from_vs(cfg, vs_val, is_halfspace=False)
        rho_val = _value_at_depth(rho_segments, mid, hs_rho if hs_rho is not None else rho_default) if rho_segments else rho_default
        if vp_val <= vs_val:
            vp_val = max(vs_val * 1.01, vp_default)
        layers.append([thickness, vp_val, vs_val, rho_val])

    hs_vp_final = hs_vp if hs_vp is not None else _vp_from_vs(cfg, hs_vs, is_halfspace=True)
    hs_rho_final = hs_rho if hs_rho is not None else _rho_from_vs(cfg, hs_vs, is_halfspace=True)
    rows = layers + [[0.0, hs_vp_final, hs_vs, hs_rho_final]]

    lines = [str(len(rows))]
    for thk, vp, vs, rho in rows:
        lines.append(" ".join(f"{val:.12g}" for val in (thk, vp, vs, rho)))
    return lines


def _run_hvf(cfg: HVFConfig, model_lines: list[str], model_path: str | Path | None = None) -> str:
    exe = cfg.exe_path
    if not exe.is_file():
        raise FileNotFoundError(f"HVf executable not found: {exe}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_file = tmp_path / "model.txt"
        serialized = "\n".join(model_lines) + "\n"
        model_file.write_text(serialized, encoding="utf-8")

        if model_path is not None:
            external_file = Path(model_path)
            external_file.parent.mkdir(parents=True, exist_ok=True)
            external_file.write_text(serialized, encoding="utf-8")

        cmd = [
            str(exe),
            "-hv",
            "-f", str(model_file),
            "-fmin", str(cfg.fmin),
            "-fmax", str(cfg.fmax),
            "-nf", str(cfg.nf),
            "-nmr", str(cfg.nmr),
            "-nml", str(cfg.nml),
            "-nks", str(cfg.nks),
        ]

        result = subprocess.run(cmd, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "HVf.exe failed (code {code}).\nCommand: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                    code=result.returncode,
                    cmd=" ".join(cmd),
                    stdout=result.stdout or "<empty>",
                    stderr=result.stderr or "<empty>",
                )
            )

        hv_path = tmp_path / "HV.dat"
        if hv_path.exists():
            return hv_path.read_text(encoding="utf-8", errors="ignore")
        if result.stdout:
            return result.stdout
        raise RuntimeError("HVf.exe produced neither HV.dat nor stdout output.")


def _parse_hv_output(hv_text: str) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in hv_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = [_float_or_none(tok) for tok in stripped.split()]
        nums = [t for t in tokens if t is not None]
        if len(nums) >= 2:
            rows.append((float(nums[0]), float(nums[1])))

    if len(rows) < 4:
        tokens = []
        for tok in hv_text.split():
            val = _float_or_none(tok)
            if val is not None and not math.isnan(val):
                tokens.append(float(val))
        if len(tokens) < 4 or len(tokens) % 2:
            raise RuntimeError("HV output too short or malformed.")
        rows = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]

    freq = np.array([r[0] for r in rows], dtype=float)
    amp = np.array([r[1] for r in rows], dtype=float)
    return freq, amp


# ---------------------------------------------------------------------------
# Aggregated output
# ---------------------------------------------------------------------------


def _write_curves(path: Path, curves: Iterable[HVCurve]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# H/V curves generated by HV_In_Batch_Modeling.py\n")
        out.write("# Each section: Model <id> (optional value) followed by Frequency[Hz] and H/V amplitude\n\n")
        for curve in curves:
            header = f"# Model {curve.model_id}"
            if curve.value is not None:
                header += f" value={curve.value:.6f}"
            out.write(header + "\n")
            out.write("#Frequency[Hz]  H/V_Amplitude\n")
            for f, a in zip(curve.frequency, curve.amplitude):
                out.write(f"{f:.6f} {a:.6f}\n")
            out.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-run HVf.exe for multiple Vs/Vp/rho models")
    parser.add_argument("--vs-file", required=True, help="Path to the Vs step-format file")
    parser.add_argument("--vp-file", required=True, help="Path to the Vp step-format file")
    parser.add_argument("--rho-file", required=True, help="Path to the rho step-format file")
    parser.add_argument("--output", required=True, help="Where to write the aggregated HV curves")
    parser.add_argument("--count", type=int, default=None, help="Number of models to process (default: all)")
    parser.add_argument("--start", type=int, default=0, help="Zero-based index to start from (default: 0)")
    parser.add_argument("--model-dir", type=str, default=None, help="Optional directory to save generated model.txt files")
    parser.add_argument("--hvf-exe", default=str(Path.cwd() / "HVf.exe"), help="Path to HVf.exe")
    parser.add_argument("--fmin", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=4.0)
    parser.add_argument("--nf", type=int, default=71)
    parser.add_argument("--nmr", type=int, default=10)
    parser.add_argument("--nml", type=int, default=10)
    parser.add_argument("--nks", type=int, default=10)
    parser.add_argument("--vp-vs-ratio-layers", type=float, default=2.5)
    parser.add_argument("--vp-vs-ratio-halfspace", type=float, default=2.0)
    parser.add_argument("--rho-mode", choices=["constant", "linear_vs"], default="constant")
    parser.add_argument("--rho-layers-value", type=float, default=1844.0)
    parser.add_argument("--rho-halfspace-value", type=float, default=2500.0)
    parser.add_argument("--rho-base", type=float, default=1600.0)
    parser.add_argument("--rho-k", type=float, default=0.5)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    vs_path = _coerce_path(args.vs_file)
    vp_path = _coerce_path(args.vp_file)
    rho_path = _coerce_path(args.rho_file)
    out_path = _coerce_path(args.output)
    model_dir = _coerce_path(args.model_dir) if args.model_dir else None

    if not vs_path.is_file():
        parser.error(f"Vs file not found: {vs_path}")
    if not vp_path.is_file():
        parser.error(f"Vp file not found: {vp_path}")
    if not rho_path.is_file():
        parser.error(f"rho file not found: {rho_path}")

    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    cfg = HVFConfig(
        exe_path=_coerce_path(args.hvf_exe),
        fmin=args.fmin,
        fmax=args.fmax,
        nf=args.nf,
        nmr=args.nmr,
        nml=args.nml,
        nks=args.nks,
        vp_vs_ratio_layers=args.vp_vs_ratio_layers,
        vp_vs_ratio_halfspace=args.vp_vs_ratio_halfspace,
        rho_mode=args.rho_mode,
        rho_layers_value=args.rho_layers_value,
        rho_halfspace_value=args.rho_halfspace_value,
        rho_base=args.rho_base,
        rho_k=args.rho_k,
    )

    vs_blocks = _read_model_blocks(vs_path, "Vs")
    vp_blocks = _read_model_blocks(vp_path, "Vp")
    rho_blocks = _read_model_blocks(rho_path, "rho")

    vp_map = {blk.model_id: blk for blk in vp_blocks}
    rho_map = {blk.model_id: blk for blk in rho_blocks}

    selected: list[ModelBlock] = []
    start_idx = max(0, args.start)
    max_models = args.count if args.count is not None else len(vs_blocks) - start_idx
    for idx, blk in enumerate(vs_blocks[start_idx:], start=start_idx):
        if blk.model_id not in vp_map or blk.model_id not in rho_map:
            raise RuntimeError(f"Model {blk.model_id} missing in Vp or rho files.")
        selected.append(blk)
        if len(selected) >= max_models:
            break

    if not selected:
        raise RuntimeError("No models selected for processing.")

    curves: list[HVCurve] = []
    for pos, vs_blk in enumerate(selected, start=1):
        model_id = vs_blk.model_id
        print(f"[INFO] Processing model {model_id} ({pos}/{len(selected)})")
        vp_blk = vp_map[model_id]
        rho_blk = rho_map[model_id]

        model_lines = _build_model_lines(cfg, vs_blk.pairs, vp_blk.pairs, rho_blk.pairs)

        model_file_path = model_dir / f"model_{pos:03d}_{model_id}.txt" if model_dir is not None else None

        hv_text = _run_hvf(cfg, model_lines, model_file_path)
        freq, amp = _parse_hv_output(hv_text)

        curves.append(HVCurve(model_id=model_id, frequency=freq, amplitude=amp, value=vs_blk.value))

    _write_curves(out_path, curves)
    print(f"[INFO] Wrote {len(curves)} curves to {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
