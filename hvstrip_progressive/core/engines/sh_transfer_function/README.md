# `sh_transfer_function` — Package Reference

Computes the **SH-wave transfer function** for vertically-propagating,
horizontally-polarized shear waves through a 1-D layered earth model.
Ported from MATLAB code by David Teague (UT Austin, 2016); physics follow
Kramer (1996), Mayne (2001), and Darendeli (2001).

---

## Package Structure

```
sh_transfer_function/
  __init__.py              ← public API (import everything from here)
  config.py                ← SHTFConfig  — all run settings
  model_io.py              ← SHTFModel   — load from .txt / .mat / arrays
  postprocess.py           ← SHTFResult  — F0 detection, Convergence Index
  runner.py                ← compute_sh_tf() — single entry point
  ── physics (low-level) ──────────────────────────────────────────
  assign_unit_wt.py        ← Mayne (2001) unit weight estimation
  mean_eff_stress.py       ← hydrostatic effective stress
  darendeli_calc.py        ← Darendeli (2001) G/Gmax and Dmin curves
  calc_wave_coeff.py       ← propagator-matrix wave coefficients
  tf_for_ground_models.py  ← orchestrates the 4 physics steps above
```

---

## Quick Start

```python
from sh_transfer_function import SHTFConfig, compute_sh_tf

config = SHTFConfig(fmin=1.0, fmax=30.0, n_samples=1024)
result = compute_sh_tf("exampl_model2.txt", config)

print(result.f0)            # [9.30]  fundamental frequency [Hz]
print(result.amplitudes)    # shape (1024, 1) — TF curve
print(result.frequencies)   # shape (1024,)   — frequency vector [Hz]
```

---

## Input Format

### Option A — Text file (`.txt`)

```
<N>                              ← number of rows (layers + half-space)
<thickness>  <Vp>  <Vs>  <density>
...
<0>          <Vp>  <Vs>  <density>   ← last row: thickness = 0 flags half-space
```

| Column | Quantity             | Units |
| ------ | -------------------- | ----- |
| 0      | Layer thickness      | m     |
| 1      | Vp                   | m/s   |
| 2      | Vs                   | m/s   |
| 3      | Density _(not used)_ | kg/m³ |

> **Note:** Density is read but ignored — unit weights are computed internally  
> via Mayne (2001) from Vs and depth.

The last layer must have **thickness = 0** to mark it as the elastic half-space.

```python
from sh_transfer_function import load_from_txt
model = load_from_txt("model.txt")
```

Custom column order (if your file differs):

```python
model = load_from_txt("model.txt", col_thickness=0, col_vp=2, col_vs=1)
```

---

### Option B — MATLAB `.mat` file

Must contain variables `depth`, `Vs`, `Vp` (column vectors, shape `(N, 1)`).

```python
from sh_transfer_function import load_from_mat
model = load_from_mat("Input.mat")
```

---

### Option C — NumPy arrays directly

```python
import numpy as np
from sh_transfer_function import load_from_arrays

depth = np.array([0.0, 2.0, 5.0, 9.0, 14.0, 20.0, 28.0, 38.0])  # m
Vs    = np.array([186,  297,  489,  646,  973, 1226, 1335, 1413]) # m/s
Vp    = np.array([369,  590,  970, 1282, 1686, 2124, 2312, 2447]) # m/s

model = load_from_arrays(depth, Vs, Vp)
```

**Layer convention:**

```
depth[0] = 0.0         ← always ground surface
           ┌──────────────────┐
 Layer 1   │ Vs[0]   Vp[0]   │  thickness = depth[1] - depth[0]
           ├──────────────────┤
 Layer 2   │ Vs[1]   Vp[1]   │  thickness = depth[2] - depth[1]
           │     ...          │
           ├──────────────────┤
 Half-space│ Vs[N-1] Vp[N-1] │  infinite thickness
           └──────────────────┘
```

---

## `SHTFConfig` Settings

```python
from sh_transfer_function import SHTFConfig

config = SHTFConfig(
    fmin      = 0.1,    # Hz — minimum frequency
    fmax      = 10.0,   # Hz — maximum frequency
    n_samples = 512,    # number of frequency points
    sampling  = "log",  # "log" or "linear"
    Dsoil     = None,   # None → Darendeli auto; float → constant % for all layers
    Drock     = 0.5,    # % — half-space damping
    d_tf      = 0,      # 0=outcrop, 'within'=top of rock, float=depth[m]
    darendeli_curvetype = 1,   # 1=mean, 2=mean+σ, 3=mean-σ
    gamma_max = 23.0,   # kN/m³ — cap on Mayne unit weights
    f0_search_fmin = None,  # Hz — F0 search window lower bound (defaults to fmin)
    f0_search_fmax = None,  # Hz — F0 search window upper bound (defaults to fmax)
    clip_tf   = 0.0,    # clip TF amplitude above this value (0 = off)
)
```

---

## `SHTFResult` Attributes

| Attribute            | Type                | Description                              |
| -------------------- | ------------------- | ---------------------------------------- |
| `.frequencies`       | `ndarray (F,)`      | Frequency vector [Hz]                    |
| `.amplitudes`        | `ndarray (F, M)`    | TF amplitude — F frequencies, M profiles |
| `.f0`                | `list[float]`       | Fundamental frequency [Hz] per profile   |
| `.peaks`             | `list[list[float]]` | All detected peaks [Hz] per profile      |
| `.convergence_index` | `float or None`     | CI from multi-method comparison          |
| `.model`             | `SHTFModel`         | Input model                              |
| `.config`            | `SHTFConfig`        | Configuration used                       |
| `.metadata`          | `dict`              | Source, n_layers, n_profiles, etc.       |

---

## Convergence Index (Multi-Method Workflow)

```python
from sh_transfer_function import compute_convergence_index

CI = compute_convergence_index(
    f0_sh  = 9.30,   # Hz — from SH transfer function
    f0_ell = 9.15,   # Hz — from Rayleigh ellipticity
    f0_dft = 9.40,   # Hz — from Diffuse Field Theory
)
# CI = 0.97  →  very good agreement
# CI < 0.90  →  investigate site conditions
```

Formula: `CI = 1 - (max(F0) - min(F0)) / mean(F0)`

| CI        | Interpretation                                                         |
| --------- | ---------------------------------------------------------------------- |
| > 0.95    | Excellent agreement — any method reliable                              |
| 0.90–0.95 | Good agreement — minor wavefield composition effect                    |
| 0.80–0.90 | Moderate divergence — check geology (gradient, inversion?)             |
| < 0.80    | High divergence — use DFT or RE preferentially; SH TF may oversimplify |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected: **42 passed** (22 physics tests + 20 API tests).
