"""
Peeling (layer stripping) utility.

Given a single HVf-format model file (N followed by N rows: thk vp vs rho),
generate a sequence of peeled models by removing the deepest finite layer
and promoting that removed layer's properties to the new half-space at each step.

Outputs a directory structure under an output base directory:

  strip/
    Step0_{k}-layer/model_Step0_{k}-layer.txt
    Step1_{k-1}-layer/model_Step1_{k-1}-layer.txt
    ...
    Step{final}_2-layer/model_Step{final}_2-layer.txt

Where k is the number of finite layers in the initial model (N-1).
"""

from pathlib import Path
from typing import List


def _strip_comment(line: str) -> str:
    return line.split('#', 1)[0].strip()


def _read_hvf_model(filepath: Path) -> List[str]:
    """Read HVf-format model file and return list of lines (including N as first).

    Raises ValueError if format is invalid.
    """
    raw_lines = filepath.read_text(encoding='utf-8', errors='ignore').splitlines()
    lines = [ln for ln in (_strip_comment(l) for l in raw_lines) if ln]
    if not lines:
        raise ValueError("Empty model file")
    try:
        n = int(lines[0].split()[0])
    except Exception as exc:
        raise ValueError("First non-empty line must be integer N") from exc
    if n <= 0:
        raise ValueError("N must be positive")
    if len(lines) < n + 1:
        raise ValueError(f"Expected {n} rows after N, found {len(lines)-1}")
    # Return only N+N rows, preserving whitespace-separated values as provided
    return [lines[0]] + lines[1:1 + n]


def _parse_rows(model_lines: List[str]) -> List[List[float]]:
    """Convert HVf model lines (N + N rows) to list[[thk, vp, vs, rho]]."""
    try:
        n = int(model_lines[0].split()[0])
    except Exception as exc:
        raise ValueError("Invalid N line") from exc
    rows: List[List[float]] = []
    for i in range(1, 1 + n):
        parts = model_lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Model row {i} has fewer than 4 columns")
        thk, vp, vs, rho = map(float, parts[:4])
        rows.append([thk, vp, vs, rho])
    if rows[-1][0] != 0.0:
        raise ValueError("Last row must be half-space (thickness=0)")
    return rows


def _to_model_lines(rows: List[List[float]]) -> List[str]:
    n = len(rows)
    out = [str(n)]
    for r in rows:
        out.append(" ".join(f"{x:.12g}" for x in r))
    return out


def generate_peel_sequence(rows: List[List[float]]) -> List[List[List[float]]]:
    """Generate list of model row-sets for each step, promoting removed layer to HS.

    rows: initial rows [[thk, vp, vs, rho], ..., [0, vp_hs, vs_hs, rho_hs]]

    Returns: list of step models (each is rows like above), from step 0 down to 2 layers.
    """
    if not rows or rows[-1][0] != 0.0:
        raise ValueError("Model must end with a half-space row")
    finite = [r for r in rows[:-1] if r[0] > 0]
    hs = rows[-1]
    if len(finite) < 1:
        # Only half-space: just return the HS-only model
        return [rows]

    steps: List[List[List[float]]] = []
    # Step 0: initial
    steps.append([*finite, hs])

    # Iteratively remove deepest finite layer, promote it to HS
    current_finite = finite[:]
    while len(current_finite) >= 1:
        # Remove last finite layer
        removed = current_finite.pop()
        if len(current_finite) == 0:
            # Terminal: 1 finite removed -> results in 1 finite? Actually none; stop when 2-layer reached previously
            break
        new_hs = [0.0, removed[1], removed[2], removed[3]]
        step_rows = [*current_finite, new_hs]
        steps.append(step_rows)
        if len(current_finite) == 1:
            # Next step would be 2-layer model already added; stop after reaching 2 layers
            break

    return steps


def write_peel_sequence(model_path: str, output_base: str) -> Path:
    """Create strip folder and write each peeled model into StepX_{k}-layer subfolders.

    model_path: path to initial HVf-format model file
    output_base: directory under which the 'strip' folder will be created

    Returns: Path to the created 'strip' directory
    """
    model_file = Path(model_path)
    out_base = Path(output_base)
    strip_dir = out_base / "strip"
    strip_dir.mkdir(parents=True, exist_ok=True)

    model_lines = _read_hvf_model(model_file)
    rows = _parse_rows(model_lines)
    sequence = generate_peel_sequence(rows)

    # Number of finite layers at step 0
    initial_finite = max(0, len(sequence[0]) - 1)

    for step_index, step_rows in enumerate(sequence):
        finite_count = max(0, len(step_rows) - 1)
        step_name = f"Step{step_index}_{finite_count}-layer"
        step_dir = strip_dir / step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"model_{step_name}.txt"
        out_path = step_dir / file_name
        out_lines = _to_model_lines(step_rows)
        out_path.write_text("\n".join(out_lines) + "\n", encoding='utf-8')

    return strip_dir


__all__ = [
    "write_peel_sequence",
    "generate_peel_sequence",
]
