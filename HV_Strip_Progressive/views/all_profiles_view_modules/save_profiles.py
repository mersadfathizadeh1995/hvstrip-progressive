"""Per-profile save functions (HV figure, Vs figure, Vs info)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from .save_helpers import build_depth_vs, save_figure_pair


SEC_COLORS = ["green", "purple", "orange", "brown", "teal"]


def save_profile_figure(
    r,
    f0: Optional[Tuple[float, float, int]],
    secondary: Sequence[Tuple[float, float, int]],
    prof_dir: Path,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    fmt: str = "png",
) -> None:
    """Save individual HV forward-curve figure (publication quality)."""
    from matplotlib.figure import Figure

    if r.freqs is None:
        return

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.plot(r.freqs, r.amps, color="royalblue", lw=2.0, label="H/V Ratio")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("H/V Amplitude Ratio", fontsize=11)
    ax.set_title(f"HV Spectral Ratio — {r.name}", fontsize=13,
                 fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    if f0:
        ax.plot(f0[0], f0[1], "*", color="firebrick", ms=16, zorder=10,
                markeredgecolor="darkred", markeredgewidth=0.8,
                label=f"f0 = {f0[0]:.2f} Hz")
        ax.axvline(f0[0], color="firebrick", ls="--", lw=0.9, alpha=0.4)

    for j, s in enumerate(secondary):
        sc = SEC_COLORS[j % len(SEC_COLORS)]
        ax.plot(s[0], s[1], "*", color=sc, ms=13, zorder=9,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"Secondary ({s[0]:.2f} Hz, A={s[1]:.2f})")
        ax.axvline(s[0], color=sc, ls=":", lw=0.8, alpha=0.4)

    ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
              edgecolor="gray")
    fig.tight_layout()
    save_figure_pair(fig, prof_dir, "hv_forward_curve", dpi, fmt)


def save_vs_figure(
    r,
    prof_dir: Path,
    dpi: int = 300,
) -> None:
    """Save individual Vs profile figure with Vs30 annotation."""
    from matplotlib.figure import Figure

    if not r.profile:
        return

    fig = Figure(figsize=(5, 7))
    ax = fig.add_subplot(111)

    depths, vs, finite, hs = build_depth_vs(r.profile)
    if hs:
        z = depths[-1] if depths else 0.0
        depths.append(z)
        vs.append(hs[0].vs)
        depths.append(z + z * 0.25)
        vs.append(hs[0].vs)

    ax.plot(vs, depths, color="teal", lw=1.8)
    ax.fill_betweenx(depths, 0, vs, alpha=0.1, color="teal")
    ax.invert_yaxis()
    ax.set_xlabel("Vs (m/s)", fontsize=10)
    ax.set_ylabel("Depth (m)", fontsize=10)
    ax.set_title(f"Vs Profile — {r.name}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    vs_max = max(vs) if vs else 500

    try:
        from ...core.vs_average import vs_average_from_profile
        res30 = vs_average_from_profile(r.profile, target_depth=30.0)
        ax.axhline(30.0, color="blue", lw=0.8, ls="-.", alpha=0.6)
        ax.annotate(f"Vs30 = {res30.vs_avg:.0f} m/s",
                    xy=(vs_max * 0.5, 30.0),
                    xytext=(0, -10), textcoords="offset points",
                    fontsize=8, color="blue", fontweight="bold")
    except Exception:
        pass

    fig.tight_layout()
    save_figure_pair(fig, prof_dir, "vs_profile", dpi, "png")


def save_vs_info(r, prof_dir: Path) -> None:
    """Save Vs30 info text file for a single profile."""
    if not r.profile:
        return
    try:
        from ...core.vs_average import vs_average_from_profile
        res = vs_average_from_profile(r.profile, target_depth=30.0)
        with open(prof_dir / "vs30_info.txt", "w") as f:
            f.write(f"Vs30_m_per_s,{res.vs_avg:.2f}\n")
            f.write(f"Target_Depth_m,{res.target_depth:.1f}\n")
            f.write(f"Actual_Depth_m,{res.actual_depth:.1f}\n")
            f.write(f"Extrapolated,{res.extrapolated}\n")
    except Exception:
        pass
