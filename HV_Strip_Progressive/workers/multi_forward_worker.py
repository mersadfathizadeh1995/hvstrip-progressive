"""MultiForwardWorker — QThread for sequential multi-profile forward computation."""
import tempfile
import os
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal


class ProfileResult:
    """Stores HV computation result and selected peaks for one profile."""

    __slots__ = ("name", "profile", "freqs", "amps", "f0",
                 "secondary_peaks", "computed", "source_path")

    def __init__(self, name, profile=None, freqs=None, amps=None,
                 f0=None, secondary_peaks=None, computed=False,
                 source_path=""):
        self.name = name
        self.profile = profile
        self.freqs = freqs
        self.amps = amps
        self.f0 = f0                    # (freq, amp, index) or None
        self.secondary_peaks = secondary_peaks or []
        self.computed = computed
        self.source_path = source_path


class MultiForwardWorker(QThread):
    """Compute HV forward curves for multiple profiles sequentially.

    Emits progress after each profile and all results when done.
    Supports optional auto-peak detection with configurable ranges.
    """

    profile_done = pyqtSignal(int, object)   # (index, ProfileResult)
    progress = pyqtSignal(str, int, int)     # (message, current, total)
    all_done = pyqtSignal(list)              # list of ProfileResult
    error = pyqtSignal(str)

    def __init__(self, profiles, config=None, engine_name="diffuse_field",
                 auto_peak_config=None, parent=None):
        """
        Parameters
        ----------
        profiles : list of (name, path_or_data, format_hint)
            From MultiInputView.get_profiles().
        config : dict
            Engine config (fmin, fmax, nf, exe_path, ...).
        engine_name : str
            Engine name for compute_hv_curve.
        auto_peak_config : dict or None
            If provided, auto-detect peaks using these settings:
            {"n_secondary": int, "ranges": [None or {"min":, "max":}],
             "min_prominence": float, "min_amplitude": float}
        """
        super().__init__(parent)
        self._profiles = profiles
        self._config = config or {}
        self._engine_name = engine_name
        self._auto_peak_config = auto_peak_config
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import numpy as np
        try:
            from hvstrip_progressive.core.hv_forward import compute_hv_curve
            from hvstrip_progressive.core.soil_profile import SoilProfile
        except ImportError as e:
            self.error.emit(f"Import error: {e}")
            return

        results = []
        total = len(self._profiles)

        for i, (name, path_or_data, fmt) in enumerate(self._profiles):
            if self._cancelled:
                break

            self.progress.emit(f"Computing {name}...", i, total)

            try:
                # Load profile depending on format
                if fmt == "results":
                    # Already computed — load from CSV
                    result = self._load_result_from_folder(name, path_or_data)
                    results.append(result)
                    self.profile_done.emit(i, result)
                    continue

                profile = self._load_profile(name, path_or_data, fmt)
                if profile is None:
                    pr = ProfileResult(name=name, computed=False)
                    results.append(pr)
                    self.profile_done.emit(i, pr)
                    continue

                # Write temp model file
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".txt", delete=False, mode="w")
                tmp.write(profile.to_hvf_format())
                tmp.close()

                try:
                    freqs, amps = compute_hv_curve(
                        tmp.name,
                        config=self._config,
                        engine_name=self._engine_name,
                    )

                    freqs_arr = np.asarray(freqs)
                    amps_arr = np.asarray(amps)

                    # Auto-detect f0 — apply primary range if configured
                    f0 = self._detect_f0(freqs_arr, amps_arr,
                                         self._auto_peak_config)

                    # Auto-detect secondary peaks if config provided
                    secondary = []
                    if self._auto_peak_config:
                        secondary = self._detect_peaks(
                            freqs_arr, amps_arr, f0, self._auto_peak_config)

                    pr = ProfileResult(
                        name=name, profile=profile,
                        freqs=freqs_arr, amps=amps_arr,
                        f0=f0, secondary_peaks=secondary,
                        computed=True,
                        source_path=path_or_data if isinstance(path_or_data, str) else "",
                    )
                finally:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass

                results.append(pr)
                self.profile_done.emit(i, pr)

            except Exception as exc:
                pr = ProfileResult(name=name, computed=False)
                results.append(pr)
                self.progress.emit(f"Error on {name}: {exc}", i, total)
                self.profile_done.emit(i, pr)

        self.progress.emit("All profiles computed.", total, total)
        self.all_done.emit(results)

    def _load_profile(self, name, path_or_data, fmt):
        """Load a SoilProfile from a path or editor data."""
        from hvstrip_progressive.core.soil_profile import SoilProfile

        if fmt == "editor" and isinstance(path_or_data, dict):
            # Editor data — construct from dict
            return path_or_data.get("profile")

        path = str(path_or_data)
        if fmt == "dinver":
            return SoilProfile.from_dinver_prefix(
                path.replace("_vs.txt", "").replace("_vs", ""),
                name=name)
        return SoilProfile.from_auto(path, name=name)

    def _detect_f0(self, freqs, amps, cfg):
        """Detect primary peak, honouring ranges[0] if configured.

        If the user specified a frequency range for the primary peak, the
        search is restricted to that range.  Otherwise global argmax is used.
        """
        import numpy as np

        rng = None
        if cfg:
            ranges = cfg.get("ranges", [])
            if ranges:
                rng = ranges[0]  # Primary peak range (index 0)

        if rng:
            fmin_r = rng.get("min", 0.0)
            fmax_r = rng.get("max", 999.0)
            mask = (freqs >= fmin_r) & (freqs <= fmax_r)
            if np.any(mask):
                masked_amps = np.where(mask, amps, -np.inf)
                idx = int(np.argmax(masked_amps))
                return (float(freqs[idx]), float(amps[idx]), idx)

        # Fallback: global maximum
        idx = int(np.argmax(amps))
        return (float(freqs[idx]), float(amps[idx]), idx)

    def _detect_peaks(self, freqs, amps, f0, cfg):
        """Detect secondary peaks using scipy.signal.find_peaks.

        Parameters
        ----------
        freqs, amps : array
        f0 : tuple (freq, amp, index)
        cfg : dict with n_secondary, ranges, min_prominence, min_amplitude

        Returns
        -------
        list of (freq, amp, index) tuples for secondary peaks
        """
        import numpy as np
        try:
            from scipy.signal import find_peaks as _find_peaks
        except ImportError:
            return []

        min_prom = cfg.get("min_prominence", 0.3)
        min_amp = cfg.get("min_amplitude", 1.5)
        n_secondary = cfg.get("n_secondary", 1)
        ranges = cfg.get("ranges", [])

        # Find all peaks in the amplitude curve
        peak_indices, properties = _find_peaks(amps, prominence=min_prom)

        # Filter by minimum amplitude
        peak_indices = [p for p in peak_indices if amps[p] >= min_amp]

        # Exclude the primary peak — use frequency-based tolerance instead
        # of hard-coded index distance.  Tolerance = 5% of f0 frequency or
        # 2× the local frequency spacing, whichever is larger.
        f0_freq = f0[0] if f0 else 0.0
        if len(freqs) > 1:
            # Average spacing near f0 in log-scale
            df = np.median(np.diff(freqs)) if len(freqs) > 1 else 0.01
            tol = max(f0_freq * 0.05, 2 * df)
        else:
            tol = f0_freq * 0.05
        peak_indices = [p for p in peak_indices
                        if abs(freqs[p] - f0_freq) > tol]

        # Apply frequency range filters for secondary peaks
        secondary = []
        for si in range(n_secondary):
            rng = ranges[si + 1] if (si + 1) < len(ranges) else None
            best = None
            best_amp = -1.0

            for p in peak_indices:
                f = float(freqs[p])
                a = float(amps[p])

                if rng:
                    fmin_r = rng.get("min", 0.0)
                    fmax_r = rng.get("max", 999.0)
                    if f < fmin_r or f > fmax_r:
                        continue

                if a > best_amp:
                    best_amp = a
                    best = (f, a, int(p))

            if best:
                secondary.append(best)
                # Remove from candidates
                peak_indices = [p for p in peak_indices if p != best[2]]

        return secondary

    @staticmethod
    def _load_result_from_folder(name, folder_path):
        """Load a previously computed result from a profile subfolder."""
        import numpy as np
        p = Path(folder_path)
        freqs, amps = None, None
        f0 = None
        secondary = []

        # Read hv_curve.csv
        csv_file = p / "hv_curve.csv"
        if csv_file.exists():
            try:
                data = np.loadtxt(str(csv_file), delimiter=",", skiprows=1)
                if data.ndim == 2 and data.shape[1] >= 2:
                    freqs = data[:, 0]
                    amps = data[:, 1]
            except Exception:
                pass

        # Read peak_info.txt
        peak_file = p / "peak_info.txt"
        if peak_file.exists():
            try:
                pairs = {}
                for line in peak_file.read_text().strip().split("\n"):
                    if "," in line:
                        k, v = line.split(",", 1)
                        pairs[k.strip()] = v.strip()

                if "f0_Frequency_Hz" in pairs:
                    f0 = (
                        float(pairs["f0_Frequency_Hz"]),
                        float(pairs.get("f0_Amplitude", 0)),
                        int(float(pairs.get("f0_Index", 0))),
                    )

                # Load secondary peaks
                i = 1
                while f"Secondary_{i}_Frequency_Hz" in pairs:
                    secondary.append((
                        float(pairs[f"Secondary_{i}_Frequency_Hz"]),
                        float(pairs.get(f"Secondary_{i}_Amplitude", 0)),
                        int(float(pairs.get(f"Secondary_{i}_Index", 0))),
                    ))
                    i += 1
            except Exception:
                pass

        return ProfileResult(
            name=name, freqs=freqs, amps=amps,
            f0=f0, secondary_peaks=secondary,
            computed=(freqs is not None),
            source_path=str(folder_path),
        )
