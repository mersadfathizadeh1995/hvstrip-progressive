"""
Soil profile data structures and I/O utilities.

Provides dataclasses for representing layered soil/rock profiles
and functions for reading/writing profiles in various formats.
"""

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .velocity_utils import VelocityConverter


@dataclass
class Layer:
    """
    Represents a single layer in a soil profile.

    Attributes
    ----------
    thickness : float
        Layer thickness in meters. Use 0 for half-space.
    vs : float
        Shear wave velocity in m/s.
    vp : Optional[float]
        Compressional wave velocity in m/s. If None, computed from nu.
    nu : Optional[float]
        Poisson's ratio. Used to compute Vp if Vp is not provided.
    density : float
        Density in kg/m3.
    is_halfspace : bool
        Whether this layer represents the half-space (infinite thickness).
    """

    thickness: float
    vs: float
    vp: Optional[float] = None
    nu: Optional[float] = None
    density: float = 2000.0
    is_halfspace: bool = False

    def compute_vp(self) -> float:
        """
        Compute Vp for this layer.

        Priority:
        1. Use provided Vp if available
        2. Compute from Vs and nu if nu is provided
        3. Use suggested nu based on Vs

        Returns
        -------
        float
            Compressional wave velocity (m/s).
        """
        if self.vp is not None:
            return self.vp
        if self.nu is not None:
            return VelocityConverter.vp_from_vs_nu(self.vs, self.nu)
        suggested_nu = VelocityConverter.suggest_nu(self.vs)
        return VelocityConverter.vp_from_vs_nu(self.vs, suggested_nu)

    def get_effective_nu(self) -> float:
        """
        Get the effective Poisson's ratio for this layer.

        Returns
        -------
        float
            Poisson's ratio (provided or computed from Vp/Vs).
        """
        if self.nu is not None:
            return self.nu
        if self.vp is not None:
            return VelocityConverter.nu_from_vp_vs(self.vp, self.vs)
        return VelocityConverter.suggest_nu(self.vs)

    def get_soil_type(self) -> str:
        """Get description of soil type based on Vs."""
        return VelocityConverter.get_soil_type_description(self.vs)

    def validate(self) -> Tuple[bool, str]:
        """
        Validate layer parameters.

        Returns
        -------
        Tuple[bool, str]
            (is_valid, message)
        """
        if self.vs <= 0:
            return False, "Vs must be positive"
        if self.thickness < 0:
            return False, "Thickness cannot be negative"
        if not self.is_halfspace and self.thickness == 0:
            return False, "Non-halfspace layer must have positive thickness"
        if self.density <= 0:
            return False, "Density must be positive"
        
        effective_vp = self.compute_vp()
        valid, msg = VelocityConverter.validate_velocities(self.vs, effective_vp)
        if not valid:
            return False, msg
        
        return True, "Valid"


def _parse_rho_model_format(rho_file: str):
    """Parse density from HVf model format (n_layers header + rows: thickness vp vs density).
    
    Returns
    -------
    Tuple[List[Tuple[float, float, float]], float]
        (segments as (d_start, d_end, density), halfspace_density)
    """
    try:
        lines = Path(rho_file).read_text(encoding="utf-8", errors="ignore").splitlines()
        # Strip comments and blanks
        data_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
        if not data_lines:
            return None, None
        
        n_layers = int(data_lines[0])
        segments = []
        hs_rho = None
        depth = 0.0
        
        for i in range(1, min(n_layers + 1, len(data_lines))):
            parts = data_lines[i].split()
            if len(parts) < 4:
                continue
            thickness = float(parts[0])
            density = float(parts[3])
            
            if thickness == 0:
                # Half-space
                hs_rho = density
            else:
                segments.append((depth, depth + thickness, density))
                depth += thickness
        
        return segments if segments else None, hs_rho
    except Exception:
        return None, None


@dataclass
class SoilProfile:
    """
    Represents a complete soil profile with multiple layers.

    Attributes
    ----------
    layers : List[Layer]
        List of layers from top to bottom.
    name : str
        Optional name for the profile.
    description : str
        Optional description.
    """

    layers: List[Layer] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the profile."""
        self.layers.append(layer)

    def remove_layer(self, index: int) -> None:
        """Remove a layer by index."""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)

    def move_layer(self, from_index: int, to_index: int) -> None:
        """Move a layer from one position to another."""
        if 0 <= from_index < len(self.layers) and 0 <= to_index < len(self.layers):
            layer = self.layers.pop(from_index)
            self.layers.insert(to_index, layer)

    def get_total_thickness(self) -> float:
        """Get total thickness of all finite layers."""
        return sum(
            layer.thickness for layer in self.layers if not layer.is_halfspace
        )

    def get_depth_to_layer(self, index: int) -> float:
        """Get depth to the top of a layer."""
        return sum(
            layer.thickness for layer in self.layers[:index] if not layer.is_halfspace
        )

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the entire profile.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list of error messages)
        """
        errors = []
        
        if not self.layers:
            errors.append("Profile must have at least one layer")
            return False, errors
        
        halfspace_count = sum(1 for layer in self.layers if layer.is_halfspace)
        if halfspace_count == 0:
            errors.append("Profile must have a half-space layer")
        elif halfspace_count > 1:
            errors.append("Profile can only have one half-space layer")
        
        if halfspace_count == 1 and not self.layers[-1].is_halfspace:
            errors.append("Half-space must be the last layer")
        
        for i, layer in enumerate(self.layers):
            valid, msg = layer.validate()
            if not valid:
                errors.append(f"Layer {i + 1}: {msg}")
        
        return len(errors) == 0, errors

    def to_hvf_format(self) -> str:
        """
        Export profile to HVf format.

        Returns
        -------
        str
            Profile in HVf format.
        """
        lines = []
        lines.append(str(len(self.layers)))
        
        for layer in self.layers:
            thickness = 0 if layer.is_halfspace else layer.thickness
            vp = layer.compute_vp()
            vs = layer.vs
            density = layer.density / 1000.0  # Convert to g/cm3
            # HVf format: thickness vp vs density
            lines.append(f"{thickness:.2f} {vp:.1f} {vs:.1f} {density:.3f}")
        
        return "\n".join(lines)

    def to_csv(self, include_computed: bool = True) -> str:
        """
        Export profile to CSV format.

        Parameters
        ----------
        include_computed : bool
            If True, include computed Vp and suggested nu columns.

        Returns
        -------
        str
            Profile in CSV format.
        """
        lines = []
        
        if include_computed:
            headers = ["thickness", "vs", "vp", "vp_computed", "nu", "nu_effective", 
                       "density", "is_halfspace", "soil_type"]
        else:
            headers = ["thickness", "vs", "vp", "nu", "density", "is_halfspace"]
        
        lines.append(",".join(headers))
        
        for layer in self.layers:
            if include_computed:
                row = [
                    f"{layer.thickness:.2f}",
                    f"{layer.vs:.1f}",
                    f"{layer.vp:.1f}" if layer.vp else "",
                    f"{layer.compute_vp():.1f}",
                    f"{layer.nu:.3f}" if layer.nu else "",
                    f"{layer.get_effective_nu():.3f}",
                    f"{layer.density:.1f}",
                    "1" if layer.is_halfspace else "0",
                    layer.get_soil_type()
                ]
            else:
                row = [
                    f"{layer.thickness:.2f}",
                    f"{layer.vs:.1f}",
                    f"{layer.vp:.1f}" if layer.vp else "",
                    f"{layer.nu:.3f}" if layer.nu else "",
                    f"{layer.density:.1f}",
                    "1" if layer.is_halfspace else "0"
                ]
            lines.append(",".join(row))
        
        return "\n".join(lines)

    @classmethod
    def from_hvf_file(cls, file_path: str) -> "SoilProfile":
        """
        Load profile from HVf format file.

        Parameters
        ----------
        file_path : str
            Path to the HVf format file.

        Returns
        -------
        SoilProfile
            Loaded profile.
        """
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        return cls.from_hvf_string(content, name=path.stem)

    @classmethod
    def from_hvf_string(cls, content: str, name: str = "") -> "SoilProfile":
        """
        Parse profile from HVf format string.

        HVf format:
        Line 1: number of layers
        Following lines: thickness vs vp density

        Parameters
        ----------
        content : str
            HVf format string.
        name : str
            Optional name for the profile.

        Returns
        -------
        SoilProfile
            Parsed profile.
        """
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        
        if not lines:
            raise ValueError("Empty profile content")
        
        num_layers = int(lines[0])
        profile = cls(name=name)
        
        for i in range(1, min(num_layers + 1, len(lines))):
            parts = lines[i].split()
            if len(parts) >= 4:
                thickness = float(parts[0])
                vp = float(parts[1])
                vs = float(parts[2])
                density = float(parts[3]) * 1000.0  # Convert from g/cm3 to kg/m3
                
                is_halfspace = thickness == 0
                
                layer = Layer(
                    thickness=thickness,
                    vs=vs,
                    vp=vp,
                    density=density,
                    is_halfspace=is_halfspace
                )
                profile.add_layer(layer)
        
        return profile

    @classmethod
    def from_csv_file(cls, file_path: str) -> "SoilProfile":
        """
        Load profile from CSV file.

        Auto-detects two CSV layouts:

        1. **Layer-table format** — columns containing ``thickness`` and ``vs``
           (case-insensitive, partial match, e.g. ``Thickness (m)``,
           ``Median Vs (m/s)``).
        2. **Depth-step format** — columns containing ``depth`` and ``vs``
           (e.g. ``Depth(m)``, ``Vs(m/s)``).  Consecutive pairs of the same
           velocity at different depths define layers.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.

        Returns
        -------
        SoilProfile
            Loaded profile.
        """
        path = Path(file_path)
        profile = cls(name=path.stem)

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            lower_headers = {h: h.lower().replace(" ", "") for h in headers}

            def _find_col(*keywords):
                """Find column whose lower/stripped name contains a keyword."""
                for kw in keywords:
                    for orig, lh in lower_headers.items():
                        if kw in lh:
                            return orig
                return None

            col_thickness = _find_col("thickness", "h(")
            col_vs = _find_col("medianvs", "vs(", "vs")
            col_depth = _find_col("depth(", "depth", "z(")
            col_vp = _find_col("vp(", "vp")
            col_nu = _find_col("nu", "poisson")
            col_density = _find_col("density", "rho(", "rho")
            col_halfspace = _find_col("halfspace")

            # ── Depth-step CSV (Depth, Vs pairs) ───────────────
            if col_depth and col_vs and not col_thickness:
                return cls._from_csv_depth_step(reader, col_depth, col_vs, path.stem)

            # ── Layer-table CSV ─────────────────────────────────
            if col_vs is None:
                raise ValueError(
                    f"CSV has no recognisable Vs column.  Headers: {headers}"
                )

            rows = list(reader)
            n_rows = len(rows)
            for row_idx, row in enumerate(rows):
                th_str = row.get(col_thickness, "0").strip() if col_thickness else "0"
                thickness = float(th_str) if th_str else 0.0
                vs = float(row[col_vs])

                vp_str = row.get(col_vp, "").strip() if col_vp else ""
                vp = float(vp_str) if vp_str else None

                nu_str = row.get(col_nu, "").strip() if col_nu else ""
                nu = float(nu_str) if nu_str else None

                den_str = row.get(col_density, "").strip() if col_density else ""
                density = float(den_str) if den_str else VelocityConverter.suggest_density(vs)

                hs_str = row.get(col_halfspace, "").strip().lower() if col_halfspace else ""
                is_last = (row_idx == n_rows - 1)
                is_halfspace = hs_str in ("1", "true", "yes") or (thickness == 0 and is_last)

                layer = Layer(
                    thickness=thickness,
                    vs=vs,
                    vp=vp,
                    nu=nu,
                    density=density,
                    is_halfspace=is_halfspace,
                )
                profile.add_layer(layer)

            # If no layer is marked half-space, mark the last one
            if profile.layers and not any(L.is_halfspace for L in profile.layers):
                profile.layers[-1].is_halfspace = True
                profile.layers[-1].thickness = 0

        return profile

    @classmethod
    def _from_csv_depth_step(
        cls, reader, col_depth: str, col_vs: str, name: str
    ) -> "SoilProfile":
        """Parse a Depth-Vs step-function CSV into a SoilProfile.

        Each pair of rows with the same Vs value at two depths defines
        a constant-velocity layer.  The last unique velocity becomes
        the half-space.
        """
        pairs = []
        for row in reader:
            d_str = row.get(col_depth, "").strip()
            v_str = row.get(col_vs, "").strip()
            if not d_str or not v_str:
                continue
            d = float(d_str)
            v = float(v_str)
            # Deduplicate consecutive rows with identical (depth, vs)
            if pairs and abs(pairs[-1][0] - d) < 1e-9 and abs(pairs[-1][1] - v) < 1e-6:
                continue
            pairs.append((d, v))

        if len(pairs) < 2:
            raise ValueError("Depth-step CSV needs at least 2 rows")

        # Extract constant-velocity segments from the step function
        layers_data = []  # (thickness, vs)
        i = 0
        while i < len(pairs) - 1:
            d_top, vs_top = pairs[i]
            # Find the bottom of this constant-velocity segment
            if i + 1 < len(pairs) and abs(pairs[i + 1][1] - vs_top) < 1e-6:
                d_bot = pairs[i + 1][0]
                thickness = d_bot - d_top
                layers_data.append((thickness, vs_top))
                i += 2
            else:
                # Single point — use next depth as boundary
                d_bot = pairs[i + 1][0]
                thickness = d_bot - d_top
                layers_data.append((thickness, vs_top))
                i += 1

        # Handle trailing single point as half-space
        if i == len(pairs) - 1:
            layers_data.append((0, pairs[i][1]))

        profile = cls(name=name)
        for idx, (th, vs) in enumerate(layers_data):
            is_hs = (idx == len(layers_data) - 1)
            # Skip zero-thickness non-halfspace layers (safety guard)
            if not is_hs and th <= 0:
                continue
            profile.add_layer(Layer(
                thickness=0 if is_hs else th,
                vs=vs,
                density=VelocityConverter.suggest_density(vs),
                is_halfspace=is_hs,
            ))
        return profile

    @classmethod
    def from_txt_file(cls, file_path: str) -> "SoilProfile":
        """
        Load profile from simple TXT file.

        Supports both HVf format and simple space/tab-delimited format.

        Parameters
        ----------
        file_path : str
            Path to the TXT file.

        Returns
        -------
        SoilProfile
            Loaded profile.
        """
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        if not lines:
            raise ValueError("Empty file")
        
        first_parts = lines[0].split()
        if len(first_parts) == 1 and first_parts[0].isdigit():
            return cls.from_hvf_file(file_path)
        
        profile = cls(name=path.stem)
        
        for line in lines:
            if line.startswith("#") or line.startswith("//"):
                continue
            
            parts = re.split(r"[,\s\t]+", line)
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 2:
                thickness = float(parts[0])
                vs = float(parts[1])
                vp = float(parts[2]) if len(parts) > 2 and parts[2] else None
                density = float(parts[3]) if len(parts) > 3 else VelocityConverter.suggest_density(vs)
                
                is_halfspace = thickness == 0
                
                layer = Layer(
                    thickness=thickness,
                    vs=vs,
                    vp=vp,
                    density=density if density > 100 else density * 1000,
                    is_halfspace=is_halfspace
                )
                profile.add_layer(layer)
        
        return profile

    @classmethod
    def from_dinver_files(
        cls,
        vs_file: str,
        vp_file: Optional[str] = None,
        rho_file: Optional[str] = None,
        name: Optional[str] = None
    ) -> "SoilProfile":
        """
        Load profile from Dinver output files (step-polyline format).

        Dinver exports velocity models as step-polyline files with format:
        value  depth (two columns, depth increases, inf for half-space)

        Parameters
        ----------
        vs_file : str
            Path to the Vs file (required).
        vp_file : str, optional
            Path to the Vp file. If None, Vp computed from Vs.
        rho_file : str, optional
            Path to the density file. If None, density estimated from Vs.
        name : str, optional
            Profile name. Defaults to Vs filename stem.

        Returns
        -------
        SoilProfile
            Loaded profile.
        """
        vs_path = Path(vs_file)
        if not vs_path.exists():
            raise FileNotFoundError(f"Vs file not found: {vs_file}")

        profile_name = name or vs_path.stem.replace("_vs", "").replace("_Vs", "")

        def parse_step_pairs(filepath: str) -> List[Tuple[float, float]]:
            """Parse step-polyline file into (value, depth) pairs."""
            content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
            pairs = []
            for line in content.splitlines():
                line = line.split('#')[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        val = float(parts[0])
                        depth_str = parts[1].lower()
                        depth = math.inf if depth_str == "inf" else float(parts[1])
                        pairs.append((val, depth))
                        if depth == math.inf:
                            break
                    except ValueError:
                        continue
            return pairs

        def segments_from_pairs(pairs: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float, float]], float]:
            """Convert pairs to segments (d_start, d_end, value) and half-space value."""
            segments = []
            i = 0
            while i + 1 < len(pairs):
                v0, d0 = pairs[i]
                v1, d1 = pairs[i + 1]
                
                if d0 != math.inf and d1 != math.inf:
                    if abs(v0 - v1) < 1e-9 and d1 > d0:
                        if d1 - d0 > 1e-12:
                            segments.append((d0, d1, v0))
                        i += 2
                        continue
                
                if d0 == d1 and abs(v0 - v1) > 1e-9:
                    i += 1
                    continue
                break

            if not segments:
                depth = 0.0
                for v, d in pairs:
                    if d == math.inf:
                        break
                    if d > 1e-12:
                        segments.append((depth, depth + d, v))
                        depth += d

            hs_val = pairs[-1][0] if pairs else 0
            for k in range(len(pairs) - 1, -1, -1):
                if pairs[k][1] == math.inf:
                    hs_val = pairs[k][0]
                    break

            return segments, hs_val

        def value_at_depth(segments: List[Tuple[float, float, float]], depth: float, default: float) -> float:
            """Get value at given depth from segments."""
            for d0, d1, v in segments:
                if d0 - 1e-12 <= depth < d1 - 1e-12:
                    return v
            return default

        vs_pairs = parse_step_pairs(vs_file)
        if not vs_pairs:
            raise ValueError("Could not parse Vs file")
        vs_segments, hs_vs = segments_from_pairs(vs_pairs)

        vp_segments, hs_vp = None, None
        if vp_file and Path(vp_file).exists():
            vp_pairs = parse_step_pairs(vp_file)
            if vp_pairs:
                vp_segments, hs_vp = segments_from_pairs(vp_pairs)

        rho_segments, hs_rho = None, None
        if rho_file and Path(rho_file).exists():
            rho_pairs = parse_step_pairs(rho_file)
            if rho_pairs:
                rho_segments, hs_rho = segments_from_pairs(rho_pairs)
            # If step-polyline parsing yielded no segments, try HVf model format
            # (n_layers header, then rows: thickness vp vs density)
            if not rho_segments:
                rho_segments, hs_rho = _parse_rho_model_format(rho_file)

        breakpoints = {0.0}
        for segs in (vs_segments, vp_segments, rho_segments):
            if segs:
                for d0, d1, _ in segs:
                    if d0 != math.inf:
                        breakpoints.add(d0)
                    if d1 != math.inf:
                        breakpoints.add(d1)
        depths = sorted(d for d in breakpoints if d >= 0)

        profile = cls(name=profile_name)

        for i in range(len(depths) - 1):
            d0, d1 = depths[i], depths[i + 1]
            thickness = d1 - d0
            if thickness < 0.01:
                continue

            mid = d0 + 0.5 * thickness
            vs_val = value_at_depth(vs_segments, mid, hs_vs)

            if vp_segments:
                vp_val = value_at_depth(vp_segments, mid, hs_vp if hs_vp else VelocityConverter.vp_from_vs_nu(vs_val, 0.35))
                if vp_val <= vs_val:
                    vp_val = vs_val * 1.5
            else:
                vp_val = None

            if rho_segments:
                rho_val = value_at_depth(rho_segments, mid, hs_rho if hs_rho else 2000)
            else:
                rho_val = VelocityConverter.suggest_density(vs_val)

            layer = Layer(
                thickness=thickness,
                vs=vs_val,
                vp=vp_val,
                density=rho_val,
                is_halfspace=False
            )
            profile.add_layer(layer)

        hs_vp_final = hs_vp if hs_vp else None
        hs_rho_final = hs_rho if hs_rho else VelocityConverter.suggest_density(hs_vs)

        hs_layer = Layer(
            thickness=0,
            vs=hs_vs,
            vp=hs_vp_final,
            density=hs_rho_final,
            is_halfspace=True
        )
        profile.add_layer(hs_layer)

        return profile

    @classmethod
    def from_dinver_prefix(cls, prefix: str, name: Optional[str] = None) -> "SoilProfile":
        """
        Load profile from Dinver files using a common prefix.

        Looks for files: prefix_vs.txt, prefix_vp.txt, prefix_rho.txt

        Parameters
        ----------
        prefix : str
            Common prefix path (e.g., "path/to/Redfield" finds Redfield_vs.txt, etc.)
        name : str, optional
            Profile name.

        Returns
        -------
        SoilProfile
            Loaded profile.
        """
        prefix_path = Path(prefix)
        
        vs_candidates = [
            prefix_path.parent / f"{prefix_path.name}_vs.txt",
            prefix_path.parent / f"{prefix_path.name}_Vs.txt",
            prefix_path.parent / f"{prefix_path.name}.vs.txt",
        ]
        
        vs_file = None
        for candidate in vs_candidates:
            if candidate.exists():
                vs_file = str(candidate)
                break
        
        if not vs_file:
            raise FileNotFoundError(f"No Vs file found for prefix: {prefix}")
        
        vp_candidates = [
            prefix_path.parent / f"{prefix_path.name}_vp.txt",
            prefix_path.parent / f"{prefix_path.name}_Vp.txt",
            prefix_path.parent / f"{prefix_path.name}.vp.txt",
        ]
        vp_file = None
        for candidate in vp_candidates:
            if candidate.exists():
                vp_file = str(candidate)
                break
        
        rho_candidates = [
            prefix_path.parent / f"{prefix_path.name}_rho.txt",
            prefix_path.parent / f"{prefix_path.name}_Rho.txt",
            prefix_path.parent / f"{prefix_path.name}.rho.txt",
            prefix_path.parent / f"{prefix_path.name}_density.txt",
        ]
        rho_file = None
        for candidate in rho_candidates:
            if candidate.exists():
                rho_file = str(candidate)
                break
        
        return cls.from_dinver_files(vs_file, vp_file, rho_file, name)

    @classmethod
    def from_excel_file(cls, file_path: str, name: Optional[str] = None) -> "SoilProfile":
        """
        Load profile from Excel (.xlsx) file.

        Parameters
        ----------
        file_path : str
            Path to .xlsx file with columns:
            Thickness (m) | Vs (m/s) | Vp (m/s) | Density (kg/m³).
            Last row is halfspace (thickness = 0, '-', or empty).
        name : str, optional
            Profile name. Defaults to file stem.

        Returns
        -------
        SoilProfile
        """
        import openpyxl

        path = Path(file_path)
        if name is None:
            name = path.stem

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active

        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue
            if all(v is None for v in row):
                continue
            rows.append(row)
        wb.close()

        if not rows:
            raise ValueError(f"No data rows found in {path.name}")

        profile = cls(name=name)
        n_rows = len(rows)

        for idx, row in enumerate(rows):
            raw_h = row[0] if len(row) > 0 else None
            raw_vs = row[1] if len(row) > 1 else None
            raw_vp = row[2] if len(row) > 2 else None
            raw_rho = row[3] if len(row) > 3 else None

            vs = float(raw_vs) if raw_vs is not None else 0.0
            vp = float(raw_vp) if raw_vp is not None else None
            density = float(raw_rho) if raw_rho is not None else None

            is_last = idx == n_rows - 1
            is_hs = False
            thickness = 0.0

            if is_last:
                if raw_h is None or str(raw_h).strip() in ('', '0', '-', '0.0'):
                    is_hs = True
                else:
                    h_val = float(raw_h)
                    if h_val <= 0:
                        is_hs = True
                    else:
                        thickness = h_val
            else:
                thickness = float(raw_h) if raw_h is not None else 0.0

            profile.add_layer(Layer(
                thickness=thickness,
                vs=vs,
                vp=vp,
                density=density,
                is_halfspace=is_hs,
            ))

        return profile

    @classmethod
    def from_auto(cls, file_path: str, name: Optional[str] = None) -> "SoilProfile":
        """Auto-detect file format and load profile.

        Supported formats (by extension):
        - ``.xlsx`` → Excel
        - ``.csv``  → CSV (header row with thickness/vs/vp/density columns)
        - ``.txt``  → tries HVf first, falls back to simple TXT
        - Dinver prefix detection when ``_vs.txt`` / ``_Vs.txt`` is found

        Parameters
        ----------
        file_path : str
            Path to the profile file.
        name : str, optional
            Profile name override.  Defaults to the file stem.

        Returns
        -------
        SoilProfile
        """
        path = Path(file_path)
        if name is None:
            name = path.stem
        ext = path.suffix.lower()

        if ext == ".xlsx":
            return cls.from_excel_file(str(path), name=name)

        if ext == ".csv":
            return cls.from_csv_file(str(path))

        # Check for Dinver convention: name ends with _vs / _Vs / _VS
        stem_lower = path.stem.lower()
        if stem_lower.endswith("_vs"):
            prefix = str(path)[: -len("_vs.txt")]
            try:
                return cls.from_dinver_prefix(prefix, name=name)
            except Exception:
                pass  # fall through to HVf / simple TXT

        # .txt or unknown → try HVf, then dinver step-polyline, then simple TXT
        if ext in (".txt", ""):
            # Peek at file to detect Dinver step-polyline format
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                is_dinver_step = False
                if lines:
                    # Dinver step files start with "# Vs" or "# Vp" or "# rho"
                    first_non_comment = ""
                    for ln in lines:
                        if ln.startswith("#"):
                            if any(kw in ln.lower() for kw in ("vs", "vp", "rho", "density")):
                                is_dinver_step = True
                                break
                            continue
                        first_non_comment = ln
                        break
                    # Also detect by presence of 'inf' in values
                    if not is_dinver_step and any("inf" in ln.lower() for ln in lines[-3:]):
                        is_dinver_step = True
                if is_dinver_step:
                    profile = cls.from_dinver_files(str(path), name=name)
                    return profile
            except Exception:
                pass

            try:
                profile = cls.from_hvf_file(str(path))
                profile.name = name
                return profile
            except Exception:
                pass
            profile = cls.from_txt_file(str(path))
            profile.name = name
            return profile

        # Last resort
        profile = cls.from_txt_file(str(path))
        profile.name = name
        return profile

    def save_hvf(self, file_path: str) -> None:
        """Save profile to HVf format file."""
        path = Path(file_path)
        path.write_text(self.to_hvf_format(), encoding="utf-8")

    def save_csv(self, file_path: str, include_computed: bool = True) -> None:
        """Save profile to CSV file."""
        path = Path(file_path)
        path.write_text(self.to_csv(include_computed), encoding="utf-8")

    def copy(self) -> "SoilProfile":
        """Create a deep copy of this profile."""
        new_profile = SoilProfile(name=self.name, description=self.description)
        for layer in self.layers:
            new_layer = Layer(
                thickness=layer.thickness,
                vs=layer.vs,
                vp=layer.vp,
                nu=layer.nu,
                density=layer.density,
                is_halfspace=layer.is_halfspace
            )
            new_profile.add_layer(new_layer)
        return new_profile


# ---------------------------------------------------------------------------
# Half-space display depth
# ---------------------------------------------------------------------------

def compute_halfspace_display_depth(
    total_finite_depth: float,
    *,
    hs_ratio: float = 0.25,
    min_extension: float = 5.0,
    max_extension: float | None = None,
) -> float:
    """Compute a proportional display thickness for the half-space layer.

    Parameters
    ----------
    total_finite_depth : float
        Sum of all finite-layer thicknesses (metres).
    hs_ratio : float
        Fraction of *total_finite_depth* to use (default 0.25 = 25 %).
    min_extension : float
        Minimum extension in metres (avoids a paper-thin half-space).
    max_extension : float or None
        Hard upper cap in metres.  ``None`` → ``total_finite_depth * 0.5``.

    Returns
    -------
    float
        Display thickness (metres) for the half-space layer.
    """
    if total_finite_depth <= 0:
        return max(min_extension, 20.0)

    cap = max_extension if max_extension is not None else total_finite_depth * 0.5
    extension = total_finite_depth * hs_ratio
    return float(max(min_extension, min(extension, cap)))
