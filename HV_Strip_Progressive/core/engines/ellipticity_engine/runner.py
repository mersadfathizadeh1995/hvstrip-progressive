"""
Core runner for the Rayleigh ellipticity engine.

Invokes Geopsy's ``gpell.exe`` through Git Bash and parses the output
into numpy arrays.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EllipticityConfig
from .model_io import format_gpell_model
from .postprocess import EllipticityResult, postprocess_curve


def _to_bash_path(win_path: str) -> str:
    """Convert a Windows path to a Git Bash (MSYS) path."""
    p = Path(win_path).resolve()
    return f"/{p.drive[0].lower()}{p.as_posix()[2:]}"


def _build_gpell_command(
    model_bash_path: str,
    config: EllipticityConfig,
) -> str:
    """Build the gpell command string for Git Bash.

    Parameters
    ----------
    model_bash_path : str
        MSYS-style path to the temporary model file.
    config : EllipticityConfig
        Engine configuration.

    Returns
    -------
    str
        Full bash command string including PATH export.
    """
    gpell_bin = str(Path(config.gpell_path).parent)
    env_prefix = f'export PATH="{_to_bash_path(gpell_bin)}:$PATH" && '

    parts = ["gpell"]
    parts += ["-n", str(config.n_samples)]
    parts += ["-min", str(config.fmin)]
    parts += ["-max", str(config.fmax)]
    parts += ["-R", str(config.n_modes)]
    parts += ["-s", config.sampling]

    if config.absolute:
        parts.append("-abs")
    if config.peak_refinement:
        parts.append("-pc")
    else:
        parts.append("-c")
    if config.love_alpha > 0:
        parts += ["-love", str(config.love_alpha)]

    parts.append(model_bash_path)
    return env_prefix + " ".join(parts)


def _get_bash_exe(git_bash_path: str) -> str:
    """Resolve the actual bash.exe from the Git installation."""
    bash_exe = str(Path(git_bash_path).parent / "usr" / "bin" / "bash.exe")
    if Path(bash_exe).exists():
        return bash_exe
    return git_bash_path


def _run_git_bash(
    git_bash_path: str,
    bash_cmd: str,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Execute a command through Git Bash.

    Parameters
    ----------
    git_bash_path : str
        Path to git-bash.exe.
    bash_cmd : str
        Full bash command string.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    subprocess.CompletedProcess
    """
    bash_exe = _get_bash_exe(git_bash_path)
    return subprocess.run(
        [bash_exe, "-c", bash_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def parse_gpell_output(
    stdout: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Parse gpell stdout into per-mode (freqs, amps) arrays.

    Parameters
    ----------
    stdout : str
        Raw stdout from gpell.

    Returns
    -------
    Dict[int, Tuple[np.ndarray, np.ndarray]]
        Mapping of mode number to (frequencies, amplitudes).
    """
    modes: Dict[int, Tuple[List[float], List[float]]] = {}
    current_mode: Optional[int] = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# Mode"):
            current_mode = int(line.split()[-1])
            modes[current_mode] = ([], [])
            continue
        if line.startswith("#"):
            continue
        if current_mode is None:
            current_mode = 0
            modes[current_mode] = ([], [])

        parts = line.split()
        if len(parts) >= 2:
            try:
                modes[current_mode][0].append(float(parts[0]))
                modes[current_mode][1].append(float(parts[1]))
            except ValueError:
                continue

    return {
        mode: (np.array(freqs), np.array(amps))
        for mode, (freqs, amps) in modes.items()
    }


def run_gpell(
    model_path: str,
    config: Optional[EllipticityConfig] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Run gpell on a model file and return raw per-mode curves.

    Parameters
    ----------
    model_path : str
        Path to the model file (HVf format).
    config : EllipticityConfig, optional
        Configuration. Uses defaults if None.

    Returns
    -------
    Dict[int, Tuple[np.ndarray, np.ndarray]]
        Mapping of mode number to (frequencies, amplitudes).

    Raises
    ------
    FileNotFoundError
        If model file, gpell, or git-bash not found.
    RuntimeError
        If gpell execution fails.
    """
    if config is None:
        config = EllipticityConfig()
    config.validate()

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_text = format_gpell_model(
        model_path,
        auto_q=config.auto_q,
        q_formula=config.q_formula,
    )

    with tempfile.TemporaryDirectory() as tdir:
        tmp_model = Path(tdir) / "model_gpell.txt"
        tmp_model.write_text(model_text, encoding="utf-8")

        bash_model_path = _to_bash_path(str(tmp_model))
        bash_cmd = _build_gpell_command(bash_model_path, config)

        try:
            result = _run_git_bash(
                config.git_bash_path, bash_cmd, config.timeout
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"gpell timed out after {config.timeout}s"
            ) from e

        if result.returncode != 0:
            raise RuntimeError(
                f"gpell failed (rc={result.returncode}):\n"
                f"  stderr: {result.stderr.strip()}\n"
                f"  stdout: {result.stdout[:500]}"
            )

        modes = parse_gpell_output(result.stdout)
        if not modes:
            raise RuntimeError(
                "gpell produced no parseable output.\n"
                f"  stdout: {result.stdout[:500]}\n"
                f"  stderr: {result.stderr[:500]}"
            )

    return modes


def compute_ellipticity(
    model_path: str,
    config: Optional[EllipticityConfig] = None,
) -> EllipticityResult:
    """Full pipeline: run gpell and return post-processed result.

    Parameters
    ----------
    model_path : str
        Path to the model file (HVf format).
    config : EllipticityConfig, optional
        Configuration. Uses defaults if None.

    Returns
    -------
    EllipticityResult
        Post-processed ellipticity curve with peaks and metadata.
    """
    if config is None:
        config = EllipticityConfig()

    modes = run_gpell(model_path, config)

    if 0 in modes:
        freqs, amps = modes[0]
    else:
        first_mode = min(modes.keys())
        freqs, amps = modes[first_mode]

    return postprocess_curve(
        freqs=freqs,
        amps=amps,
        clip_factor=config.clip_factor,
        all_modes=modes,
        model_path=model_path,
        config=config,
    )


__all__ = [
    "run_gpell",
    "compute_ellipticity",
    "parse_gpell_output",
]
