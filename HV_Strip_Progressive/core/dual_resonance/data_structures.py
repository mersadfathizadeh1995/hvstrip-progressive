"""Data structures for dual-resonance analysis."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DualResonanceResult:
    """Result of dual-resonance extraction for one profile."""

    profile_name: str
    profile_path: str
    success: bool

    # Layer geometry
    n_layers: int = 0
    total_depth: float = 0.0
    layer_thicknesses: List[float] = field(default_factory=list)
    layer_vs: List[float] = field(default_factory=list)

    # Deep resonance (original full model — Step 0 by default)
    f0: float = 0.0
    a0: float = 0.0
    f0_theoretical: float = 0.0

    # Shallow resonance (after removing deepest layer — Step 1 by default)
    f1: float = 0.0
    a1: float = 0.0
    f1_theoretical: float = 0.0

    # Per-step peak tracking
    freq_per_step: List[float] = field(default_factory=list)
    amp_per_step: List[float] = field(default_factory=list)

    # Step names for cross-referencing with wizard peak data
    step_names: List[str] = field(default_factory=list)

    # Which step pair was used for (f0, f1)
    step_pair: tuple = (0, 1)

    # Derived metrics
    freq_ratio: float = 0.0       # f1 / f0
    max_freq_shift: float = 0.0   # largest |Δf| between consecutive steps
    controlling_step: int = 0     # step index with largest shift
    separation_success: bool = False

    error_message: str = ""


@dataclass
class BatchDualResonanceStats:
    """Aggregate statistics from a batch of dual-resonance analyses."""

    n_profiles: int = 0
    n_successful: int = 0
    success_rate: float = 0.0

    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0

    f1_mean: float = 0.0
    f1_std: float = 0.0
    f1_min: float = 0.0
    f1_max: float = 0.0

    freq_ratio_mean: float = 0.0
    freq_ratio_std: float = 0.0

    f0_theoretical_correlation: float = 0.0
    f1_theoretical_correlation: float = 0.0

    separation_success_rate: float = 0.0

    mean_freq_shift_per_step: List[float] = field(default_factory=list)
    std_freq_shift_per_step: List[float] = field(default_factory=list)
