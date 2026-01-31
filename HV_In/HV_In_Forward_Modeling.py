import os
import re
import sys
import math
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1) User-defined paths and parameters (kept like your original script)
# ------------------------------------------------------------------------------
exe_path    = r"D:\Research\Analysis_tools\HV_In\HVf.exe"
models_file = r"D:\Runs\Independence_Plant\Phase_2\Processed\Dinver\Figures\Hv_in\Indep_Powerplant_B19_vs.txt"
# Optional: provide a prefix like "Run_55_B6" (with or without directory) to auto-load
# Run_55_B6.vs.txt, Run_55_B6.vp.txt, Run_55_B6.rho.txt from the same directory.
# If empty, the script uses 'models_file' as before.
model_prefix = r"D:\Runs\Independence_Plant\Phase_2\Processed\Dinver\Figures\Hv_in\Indep_Powerplant_B19"  # e.g., r"D:\Runs\...\Run_55_B6" or just "Run_55_B6"
output_file = r"D:\Runs\Independence_Plant\Phase_2\Processed\Dinver\Figures\Hv_in\HV_Run_06.txt"

# HVf.exe command-line parameters:
fmin = "1"
fmax = "10"
nf   = "71"
nmr  = "10"
nml  = "10"
nks  = "10"

# Plot settings (new)
plot_file = r"D:\Runs\Independence_Plant\B_2\Processed\Dinver\Figures\Run_48\Vs_Profile\new\Hv_in"
show_plot = True  # set True to display the plot interactively
hide_y_axis_values = False  # set True to hide Y-axis tick labels
y_axis_scale = "linear"  # "linear" or "log"
x_axis_scale = "log"  # "linear" or "log"

# VS->(Vp,rho) defaults for formats missing them (e.g., the #Vs two-column file)
vp_vs_ratio_layers   = 2.5     # Vp = ratio * Vs for layers
vp_vs_ratio_halfspace= 2.0     # Vp = ratio * Vs for the half-space
rho_mode             = "constant"  # "constant" or "linear_vs"
rho_layers_value     = 1844.0  # used if rho_mode == "constant"
rho_halfspace_value  = 2500.0  # typical rock half-space
rho_base             = 1600.0  # if rho_mode == "linear_vs": rho = rho_base + rho_k * Vs
rho_k                = 0.5

# ------------------------------------------------------------------------------
# 2) Helpers
# ------------------------------------------------------------------------------

def _strip_comment(s: str) -> str:
    return s.split('#', 1)[0].strip()

def _float_or_none(tok: str):
    try:
        if tok.lower() == "inf":
            return math.inf
        return float(tok)
    except Exception:
        return None

def _is_single_integer_line(line: str) -> bool:
    core = _strip_comment(line)
    if not core:
        return False
    parts = core.split()
    if len(parts) != 1:
        return False
    try:
        int(parts[0])
        return True
    except ValueError:
        return False

def _rho_from_vs(vs: float, is_halfspace: bool=False) -> float:
    if rho_mode == "constant":
        return rho_halfspace_value if is_halfspace else rho_layers_value
    # linear_vs
    return max(1000.0, rho_base + rho_k * vs)

def _vp_from_vs(vs: float, is_halfspace: bool=False) -> float:
    ratio = vp_vs_ratio_halfspace if is_halfspace else vp_vs_ratio_layers
    vp = ratio * vs
    return max(vp, vs * 1.01)  # ensure Vp > Vs

def _same(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(abs_tol, rel_tol * max(1.0, abs(a), abs(b)))

def _mostly_increasing(arr, tolerance_fraction: float = 0.8) -> bool:
    try:
        arr = list(arr)
        if len(arr) < 3:
            return False
        diffs = [arr[i+1] - arr[i] for i in range(len(arr)-1)]
        positive = sum(1 for d in diffs if d >= -1e-12)
        return (positive / max(1, len(diffs))) >= tolerance_fraction
    except Exception:
        return False

def _resolve_plot_save_path(target_path: str, default_name: str = "HV_curve.png") -> str:
    p = Path(target_path)
    # If path ends with a known image extension, use as-is; otherwise, treat as directory or base name
    img_exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    if p.suffix.lower() in img_exts:
        final_path = p
    else:
        # If it's an existing directory or a path without extension, append default_name
        final_path = p / default_name
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return str(final_path)

# ------------------------------------------------------------------------------
# 3) VS format A: “N + N lines” (e.g., B2_Run25_best.txt)
#    Expected: after comments, first integer line = N, followed by N numeric rows.
#    Columns: thickness  Vp  Vs  rho  (≥3 numeric cols; Vs is the 3rd)
# ------------------------------------------------------------------------------
def parse_vs_format_A(filepath: str):
    raw = Path(filepath).read_text(encoding="utf-8", errors="ignore").splitlines()

    N = None
    idxN = None
    for i, ln in enumerate(raw):
        if _is_single_integer_line(ln):
            N = int(_strip_comment(ln))
            idxN = i
            break
    if N is None:
        return None  # not this format

    rows = []
    j = idxN + 1
    while j < len(raw) and len(rows) < N:
        core = _strip_comment(raw[j])
        if core:
            toks = core.split()
            nums = []
            for t in toks:
                v = _float_or_none(t)
                if v is not None and not math.isnan(v):
                    nums.append(v)
            if len(nums) >= 3:
                rows.append(nums)
        j += 1

    if len(rows) != N:
        raise RuntimeError(f"Format A: N={N} but found {len(rows)} numeric lines after it.")

    # Ensure 4 columns: [thk, Vp, Vs, rho]
    norm_rows = []
    for k, nums in enumerate(rows):
        thk = float(nums[0])  # by convention in these files
        if len(nums) >= 4:
            vp, vs, rho = float(nums[1]), float(nums[2]), float(nums[3])
        else:
            # assume [thk, vp?, vs] -> derive missing fields
            vs = float(nums[2])
            vp = _vp_from_vs(vs, is_halfspace=(k == N-1 and thk == 0))
            rho = _rho_from_vs(vs, is_halfspace=(k == N-1 and thk == 0))
        norm_rows.append([thk, vp, vs, rho])

    # Build model lines for HVf: first line N, then rows
    model_lines = [str(N)] + [" ".join(f"{x:.12g}" for x in r) for r in norm_rows]
    return model_lines

# ------------------------------------------------------------------------------
# 4) VS format B: “# Vs” two-column step profile (Vs, depth) ending with depth=inf
#    Example: run_55_vs.txt
# ------------------------------------------------------------------------------
def parse_vs_format_B(filepath: str):
    raw = Path(filepath).read_text(encoding="utf-8", errors="ignore").splitlines()

    # Find the "# Vs" marker
    start = None
    for i, ln in enumerate(raw):
        if ln.strip().lower().startswith("# vs"):
            start = i + 1
            break
    if start is None:
        return None  # not this format

    pairs = []
    for ln in raw[start:]:
        core = _strip_comment(ln)
        if not core:
            continue
        toks = core.split()
        if len(toks) < 2:
            continue
        vs = _float_or_none(toks[0])
        dz = _float_or_none(toks[1])
        if vs is None or dz is None:
            continue
        pairs.append((float(vs), dz))
        # stop at first infinite depth row
        if dz is math.inf:
            break

    if not pairs or pairs[-1][1] is not math.inf:
        raise RuntimeError("Format B: did not find a terminal depth = inf line.")

    # Convert the step polyline into layers:
    # Typical sequence: (Vs, d0), (Vs, d1)  -> horizontal (layer thickness = d1-d0)
    # then (Vs2, d1) -> vertical (ignore), then (Vs2, d2) -> next horizontal, etc.
    layers = []  # list of (thickness, vp, vs, rho)
    i = 0
    while i + 1 < len(pairs):
        vs0, d0 = pairs[i]
        vs1, d1 = pairs[i + 1]

        # horizontal segment (same Vs, increasing depth)
        if (d0 not in (None, math.inf)) and (d1 not in (None, math.inf)) and _same(vs0, vs1) and d1 > d0:
            thk = d1 - d0
            vp  = _vp_from_vs(vs0, is_halfspace=False)
            rho = _rho_from_vs(vs0, is_halfspace=False)
            if thk > 1e-12:
                layers.append([thk, vp, vs0, rho])
            i += 2
            continue

        # vertical edge (same depth, Vs changes) -> skip one step
        if d0 == d1 and not _same(vs0, vs1):
            i += 1
            continue

        # reached inf (half-space) or odd shape -> break to handle HS
        break

    # Fallback: if no horizontal segments recognized, treat second column as thickness
    if not layers:
        finite = [(float(vs), float(d)) for (vs, d) in pairs if d not in (None, math.inf)]
        for vs, thk in finite:
            if thk > 1e-12:
                layers.append([thk, _vp_from_vs(vs, is_halfspace=False), vs, _rho_from_vs(vs, is_halfspace=False)])

    # Half-space: take the Vs value immediately before depth=inf
    # Walk back from the end to find last finite-depth pair and its Vs
    hs_vs = None
    for k in range(len(pairs) - 1, -1, -1):
        vs_k, d_k = pairs[k]
        if d_k is math.inf and k > 0:
            # use the Vs of the previous point if it shares the same Vs (typical)
            hs_vs = pairs[k - 1][0]
            break
    if hs_vs is None:
        # fallback: use the last seen Vs in layers or the final token
        hs_vs = layers[-1][2] if layers else pairs[-1][0]

    hs_vp  = _vp_from_vs(hs_vs, is_halfspace=True)
    hs_rho = _rho_from_vs(hs_vs, is_halfspace=True)

    N = len(layers) + 1  # +1 for half-space
    model_rows = layers + [[0.0, hs_vp, hs_vs, hs_rho]]
    model_lines = [str(N)] + [" ".join(f"{x:.12g}" for x in r) for r in model_rows]
    return model_lines

# ------------------------------------------------------------------------------
# 5) Auto-detect VS format and normalize to HVf model-lines
# ------------------------------------------------------------------------------
def parse_vs_any(filepath: str):
    # Try A (N + rows)
    a = parse_vs_format_A(filepath)
    if a is not None:
        return a
    # Try B (# Vs two-column step to inf)
    b = parse_vs_format_B(filepath)
    if b is not None:
        return b
    raise RuntimeError("Could not auto-detect VS format. Expected 'N + rows' OR '# Vs' two-column step format.")

# ------------------------------------------------------------------------------
# 5a) Load Vs/Vp/rho from a prefix: *.vs.txt, *.vp.txt, *.rho.txt
#      Combine into layers with Vp>Vs and thicknesses from depth steps.
# ------------------------------------------------------------------------------
def _read_step_pairs(filepath: str, label: str):
    if not Path(filepath).exists():
        return None
    raw = Path(filepath).read_text(encoding="utf-8", errors="ignore").splitlines()
    pairs = []
    for ln in raw:
        core = _strip_comment(ln)
        if not core:
            continue
        toks = core.split()
        nums = []
        for t in toks:
            v = _float_or_none(t)
            if v is not None and not math.isnan(v):
                nums.append(v)
        if len(nums) >= 2:
            val = float(nums[0])
            depth_or_thk = nums[1]
            pairs.append((val, depth_or_thk))
            if depth_or_thk is math.inf:
                break
    if not pairs:
        return None
    # Ensure terminal inf exists; if missing, add it using last value
    if pairs[-1][1] is not math.inf:
        pairs.append((pairs[-1][0], math.inf))
    return pairs

def _segments_from_pairs(pairs):
    # Convert pairs (value, depth) with step polyline representation
    # into horizontal segments (d_start, d_end, value).
    segments = []
    i = 0
    while i + 1 < len(pairs):
        v0, d0 = pairs[i]
        v1, d1 = pairs[i + 1]
        if (d0 not in (None, math.inf)) and (d1 not in (None, math.inf)) and _same(v0, v1) and d1 > d0:
            d_start = float(d0)
            d_end   = float(d1)
            if d_end - d_start > 1e-12:
                segments.append((d_start, d_end, float(v0)))
            i += 2
            continue
        if d0 == d1 and not _same(v0, v1):
            i += 1
            continue
        break

    # Fallback: interpret as (value, thickness)
    if not segments:
        depth = 0.0
        for v, thk in pairs:
            if thk is math.inf:
                break
            thk = float(thk)
            if thk > 1e-12:
                segments.append((depth, depth + thk, float(v)))
                depth += thk

    # Determine half-space value from terminal inf
    hs_val = None
    for k in range(len(pairs) - 1, -1, -1):
        v_k, d_k = pairs[k]
        if d_k is math.inf:
            hs_val = pairs[k - 1][0] if k > 0 else v_k
            break
    if hs_val is None:
        hs_val = segments[-1][2] if segments else pairs[-1][0]

    # Normalize starts: ensure first segment starts at 0
    if segments and segments[0][0] > 1e-12:
        d0, d1, v = segments[0]
        segments[0] = (0.0, d1, v)

    return segments, float(hs_val)

def _value_at_depth(segments, depth: float, default_value: float) -> float:
    if not segments:
        return default_value
    for d0, d1, v in segments:
        if depth >= d0 - 1e-12 and depth < d1 - 1e-12:
            return v
    if depth < segments[0][0]:
        return segments[0][2]
    return default_value

def parse_triplet_from_prefix(prefix: str):
    # Resolve prefix path. If relative (no directory), place next to models_file or CWD
    pref_path = Path(prefix)
    if not pref_path.is_absolute():
        base_dir = Path(models_file).parent if models_file else Path.cwd()
        pref_path = base_dir / pref_path

    vs_path  = Path(str(pref_path) + ".vs.txt")
    vp_path  = Path(str(pref_path) + ".vp.txt")
    rho_path = Path(str(pref_path) + ".rho.txt")

    if not vs_path.exists():
        print(f"[WARN] Vs file not found for prefix: {vs_path}")
        return None

    print(f"[INFO] Reading Vs from: {vs_path}")
    vs_pairs = _read_step_pairs(str(vs_path), "Vs")
    if not vs_pairs:
        raise RuntimeError("Vs file exists but could not be parsed.")
    vs_segments, hs_vs = _segments_from_pairs(vs_pairs)

    vp_segments = None
    hs_vp = None
    if vp_path.exists():
        print(f"[INFO] Reading Vp from: {vp_path}")
        vp_pairs = _read_step_pairs(str(vp_path), "Vp")
        if vp_pairs:
            vp_segments, hs_vp = _segments_from_pairs(vp_pairs)
    else:
        print("[INFO] Vp file missing; will derive from Vs using ratios.")

    rho_segments = None
    hs_rho = None
    if rho_path.exists():
        print(f"[INFO] Reading rho from: {rho_path}")
        rho_pairs = _read_step_pairs(str(rho_path), "rho")
        if rho_pairs:
            rho_segments, hs_rho = _segments_from_pairs(rho_pairs)
    else:
        print("[INFO] rho file missing; will derive from Vs using configured mode.")

    # Build union of depth breakpoints
    bps = {0.0}
    for segs in (vs_segments, vp_segments, rho_segments):
        if segs:
            for d0, d1, _ in segs:
                if d0 not in (None, math.inf):
                    bps.add(float(d0))
                if d1 not in (None, math.inf):
                    bps.add(float(d1))
    depths = sorted(x for x in bps if x >= 0)
    layers = []
    for i in range(len(depths) - 1):
        d0, d1 = depths[i], depths[i + 1]
        thk = d1 - d0
        if thk <= 1e-12:
            continue
        mid = d0 + 0.5 * thk
        vs_val = _value_at_depth(vs_segments, mid, hs_vs)
        vp_val = _value_at_depth(vp_segments, mid, hs_vp if hs_vp is not None else _vp_from_vs(vs_val, is_halfspace=False))
        rho_val = _value_at_depth(rho_segments, mid, hs_rho if hs_rho is not None else _rho_from_vs(vs_val, is_halfspace=False))
        if vp_val <= vs_val:
            vp_val = max(vs_val * 1.01, _vp_from_vs(vs_val, is_halfspace=False))
        layers.append([thk, vp_val, vs_val, rho_val])

    # Append half-space row
    hs_vp_final = hs_vp if hs_vp is not None else _vp_from_vs(hs_vs, is_halfspace=True)
    hs_rho_final = hs_rho if hs_rho is not None else _rho_from_vs(hs_vs, is_halfspace=True)
    rows = layers + [[0.0, hs_vp_final, hs_vs, hs_rho_final]]
    N = len(rows)
    model_lines = [str(N)] + [" ".join(f"{x:.12g}" for x in r) for r in rows]
    return model_lines

# ------------------------------------------------------------------------------
# 6) Run HVf.exe and read HV.dat
# ------------------------------------------------------------------------------
def run_hvf_for_model(model_lines):
    with tempfile.TemporaryDirectory() as tdir:
        model_path = Path(tdir, "model.txt")
        model_path.write_text("\n".join(model_lines) + "\n", encoding="utf-8")

        cmd = [
            exe_path,
            "-hv",
            "-f", str(model_path),
            "-fmin", fmin,
            "-fmax", fmax,
            "-nf",   nf,
            "-nmr",  nmr,
            "-nml",  nml,
            "-nks",  nks,
        ]
        try:
            res = subprocess.run(
                cmd, cwd=tdir, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"HVf.exe failed.\nCommand: {' '.join(cmd)}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
            )

        hv_path = Path(tdir, "HV.dat")
        if not hv_path.exists():
            # Some builds might print to stdout; try that
            if res.stdout.strip():
                return res.stdout
            raise RuntimeError("HV.dat not found and no stdout returned by HVf.exe.")

        return hv_path.read_text(encoding="utf-8", errors="ignore")

# ------------------------------------------------------------------------------
# 7) Parse HV.dat to arrays (supports single-line or two-column)
# ------------------------------------------------------------------------------
def parse_hv_text(hv_text: str):
    rows = []
    for ln in hv_text.splitlines():
        core = _strip_comment(ln)
        if not core:
            continue
        nums = []
        for t in core.split():
            v = _float_or_none(t)
            if v is not None and not math.isnan(v):
                nums.append(v)
        if len(nums) >= 2:
            rows.append((float(nums[0]), float(nums[1])))  # initial assumption: first two numeric columns

    # Fallback: single-line or odd formatting -> parse all tokens as pairs
    if len(rows) < 4:
        tokens = []
        for ln in hv_text.split():
            v = _float_or_none(ln)
            if v is not None and not math.isnan(v):
                tokens.append(float(v))
        if len(tokens) >= 4 and len(tokens) % 2 == 0:
            rows = [(tokens[i], tokens[i+1]) for i in range(0, len(tokens), 2)]

    if len(rows) < 4:
        raise RuntimeError("HV output too short or malformed.")

    col1 = np.array([r[0] for r in rows], dtype=float)
    col2 = np.array([r[1] for r in rows], dtype=float)

    # Heuristic: figure out which column is frequency vs amplitude
    try:
        fmin_f = float(fmin)
        fmax_f = float(fmax)
    except Exception:
        fmin_f, fmax_f = None, None

    c1_inc = _mostly_increasing(col1)
    c2_inc = _mostly_increasing(col2)

    # If exactly one column is (mostly) increasing -> treat that as frequency
    if c1_inc and not c2_inc:
        freqs, amps = col1, col2
    elif c2_inc and not c1_inc:
        freqs, amps = col2, col1
    else:
        # Fall back to original assumption
        freqs, amps = col1, col2

    return freqs, amps

def sanity_check_hv(freqs: np.ndarray, amps: np.ndarray):
    warnings = []
    if len(freqs) != len(amps) or len(freqs) < 4:
        warnings.append("HV arrays length mismatch or too short.")
    try:
        fmin_f = float(fmin)
        fmax_f = float(fmax)
        if not (np.nanmin(freqs) >= 0):
            warnings.append("Negative frequencies detected (unexpected).")
        if (np.nanmin(freqs) > fmin_f * 1.1) or (np.nanmax(freqs) < fmax_f * 0.9):
            warnings.append("Frequency range does not match requested [fmin,fmax].")
    except Exception:
        pass
    if np.nanmax(amps) > 50:
        warnings.append("Very large H/V amplitude (>50). Possible column swap or model issue.")
    if np.any(amps < 0):
        warnings.append("Negative H/V amplitudes found (unexpected).")
    for w in warnings:
        print(f"[WARN] {w}")

# ------------------------------------------------------------------------------
# 8) Plot HV and annotate the global peak
# ------------------------------------------------------------------------------
def plot_hv_with_peak(freqs, amps, plot_path: str, title="H/V Curve main Peak"):
    idx = int(np.argmax(amps))
    f_peak = freqs[idx]
    a_peak = float(amps[idx])
    use_log_y = str(y_axis_scale).lower().startswith("log")
    use_log_x = str(x_axis_scale).lower().startswith("log")

    # Safeguard amplitudes for log scale (no zeros/negatives)
    if use_log_y:
        eps = 1e-6
        amps_safe = np.maximum(amps, eps)
        a_peak_safe = max(a_peak, eps)
    else:
        amps_safe = amps
        a_peak_safe = a_peak

    # Safeguard frequencies if using log-x
    if use_log_x:
        epsx = 1e-9
        freqs_safe = np.maximum(freqs, epsx)
        f_peak_safe = max(float(f_peak), epsx)
    else:
        freqs_safe = freqs
        f_peak_safe = float(f_peak)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(freqs_safe, amps_safe, linewidth=2)
    plt.scatter([f_peak_safe], [a_peak_safe], s=60, zorder=3)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("H/V Amplitude")
    plt.grid(True, alpha=0.3)

    # Smart annotation placement: choose a nearby location with available space
    x_min = float(np.nanmin(freqs_safe))
    x_max = float(np.nanmax(freqs_safe))

    if use_log_y:
        y_min = float(np.nanmin(amps_safe))
        y_max = float(np.nanmax(amps_safe))
        y_rng_log = max(1e-12, math.log10(y_max) - math.log10(y_min))
        # Horizontal placement (linear or log-x)
        if use_log_x:
            x_rng_log = max(1e-12, math.log10(x_max) - math.log10(x_min))
            go_right = (math.log10(x_max) - math.log10(f_peak_safe)) >= (math.log10(f_peak_safe) - math.log10(x_min))
            dx_log = 0.12 * x_rng_log
            x_text = f_peak_safe * (10 ** (dx_log) if go_right else 10 ** (-dx_log))
            # Clamp inside
            x_low = x_min * (10 ** (0.06 * x_rng_log))
            x_high = x_max / (10 ** (0.06 * x_rng_log))
            x_text = min(max(x_text, x_low), x_high)
        else:
            x_rng = max(1e-12, x_max - x_min)
            go_right = (x_max - f_peak_safe) >= (f_peak_safe - x_min)
            dx = 0.12 * x_rng
            x_text = f_peak_safe + (dx if go_right else -dx)
            x_text = min(max(x_text, x_min + 0.06 * x_rng), x_max - 0.06 * x_rng)
        above_space_log = math.log10(y_max) - math.log10(a_peak_safe)
        below_space_log = math.log10(a_peak_safe) - math.log10(y_min)
        place_below = below_space_log >= max(0.12 * y_rng_log, above_space_log)
        dy_log = 0.18 * y_rng_log
        y_text = a_peak_safe * (10 ** (-dy_log) if place_below else 10 ** (dy_log))
        # Clamp Y inside
        y_low = y_min * (10 ** (0.08 * y_rng_log))
        y_high = y_max / (10 ** (0.10 * y_rng_log))
        y_text = min(max(y_text, y_low), y_high)
        ha = "left" if go_right else "right"
        va = "top" if place_below else "bottom"
    else:
        y_min = float(np.nanmin(amps_safe))
        y_max = float(np.nanmax(amps_safe))
        y_rng = max(1e-12, y_max - y_min)
        if use_log_x:
            x_rng_log = max(1e-12, math.log10(x_max) - math.log10(x_min))
            go_right = (math.log10(x_max) - math.log10(f_peak_safe)) >= (math.log10(f_peak_safe) - math.log10(x_min))
            dx_log = 0.12 * x_rng_log
            x_text = f_peak_safe * (10 ** (dx_log) if go_right else 10 ** (-dx_log))
            x_low = x_min * (10 ** (0.06 * x_rng_log))
            x_high = x_max / (10 ** (0.06 * x_rng_log))
            x_text = min(max(x_text, x_low), x_high)
        else:
            x_rng = max(1e-12, x_max - x_min)
            go_right = (x_max - f_peak_safe) >= (f_peak_safe - x_min)
            dx = 0.12 * x_rng
            x_text = f_peak_safe + (dx if go_right else -dx)
            x_text = min(max(x_text, x_min + 0.06 * x_rng), x_max - 0.06 * x_rng)
        above_space = y_max - a_peak_safe
        below_space = a_peak_safe - y_min
        place_below = below_space >= max(0.12 * y_rng, above_space)
        dy = 0.18 * y_rng
        y_text = a_peak_safe - dy if place_below else a_peak_safe + dy
        y_text = min(max(y_text, y_min + 0.08 * y_rng), y_max - 0.10 * y_rng)
        ha = "left" if go_right else "right"
        va = "top" if place_below else "bottom"

    plt.annotate(
        f"Peak: {f_peak:.3f} Hz",
        xy=(f_peak_safe, a_peak_safe),
        xytext=(x_text, y_text),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=9,
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.3", alpha=0.9)
    )

    ax = plt.gca()
    ax.tick_params(axis='y', labelleft=(not hide_y_axis_values))
    # Add a little vertical margin to reduce crowding near the title and top border
    ax.margins(y=0.06)
    if use_log_y:
        ax.set_yscale('log')
    # Add a small horizontal margin and set log-x if requested
    ax.margins(x=0.04)
    if use_log_x:
        ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close()

# ------------------------------------------------------------------------------
# 9) Main
# ------------------------------------------------------------------------------
def main():
    # Prefer triplet from prefix if provided
    model_lines = None
    if model_prefix:
        print("[INFO] Using model prefix to read Vs/Vp/rho triplet:", model_prefix)
        try:
            model_lines = parse_triplet_from_prefix(model_prefix)
        except Exception as e:
            print("[WARN] Could not build model from prefix:", str(e))

    if model_lines is None:
        print("[INFO] Reading VS file:", models_file)
        model_lines = parse_vs_any(models_file)
    # Model sanity checks
    try:
        # Expect first line is N, followed by N rows of: thk vp vs rho
        N = int(model_lines[0])
        if N <= 0:
            print("[WARN] Model N <= 0.")
        thks = []
        vps = []
        vss = []
        rhos = []
        for raw in model_lines[1:]:
            parts = [float(x) for x in raw.split()[:4]]
            if len(parts) < 4:
                continue
            thk, vp, vs, rho = parts
            thks.append(thk)
            vps.append(vp)
            vss.append(vs)
            rhos.append(rho)
        if thks:
            for i, thk in enumerate(thks[:-1]):
                if thk <= 0:
                    print(f"[WARN] Non-positive thickness at layer {i+1}: {thk} m")
            if thks[-1] != 0:
                print("[WARN] Last row should be half-space with thickness 0.")
        for i, (vp, vs) in enumerate(zip(vps, vss), start=1):
            if vp <= vs:
                print(f"[WARN] Vp <= Vs at row {i} (Vp={vp}, Vs={vs}).")
            if vs < 50 or vs > 6000:
                print(f"[WARN] Suspicious Vs at row {i}: {vs} m/s")
        for i, rho in enumerate(rhos, start=1):
            if rho < 900 or rho > 3500:
                print(f"[WARN] Suspicious density at row {i}: {rho} kg/m^3")
    except Exception:
        print("[WARN] Could not perform full model sanity check.")
    # Concise debug preview of the model
    try:
        n_rows = int(model_lines[0])
    except Exception:
        n_rows = len(model_lines) - 1
    print(f"[DEBUG] Model rows (including half-space): N={n_rows}")
    preview_count = min(3, max(0, len(model_lines) - 1))
    if preview_count > 0:
        print("[DEBUG] First rows:")
        for l in model_lines[1:1+preview_count]:
            print("  " + l)
    print("[DEBUG] Half-space row:")
    print("  " + model_lines[-1])

    print("[INFO] Running HVf.exe …")
    hv_text = run_hvf_for_model(model_lines)

    print("[INFO] Parsing HV output …")
    freqs, amps = parse_hv_text(hv_text)
    sanity_check_hv(freqs, amps)

    # Write HV curve to text
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("#Frequency[Hz]  H/V_Amplitude\n")
        for f, a in zip(freqs, amps):
            out.write(f"{f:.6f} {a:.6f}\n")
    print(f"[INFO] H/V curve written to:\n  {output_file}")

    # Plot + annotate peak
    print("[INFO] Creating plot with peak annotation …")
    plot_out = _resolve_plot_save_path(plot_file)
    plot_hv_with_peak(freqs, amps, plot_out)
    print(f"[INFO] Plot saved to:\n  {plot_out}")

    # Console summary
    pidx = int(np.argmax(amps))
    print(f"[INFO] Peak: f = {freqs[pidx]:.6f} Hz, H/V = {amps[pidx]:.6f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", str(e))
        sys.exit(1)
