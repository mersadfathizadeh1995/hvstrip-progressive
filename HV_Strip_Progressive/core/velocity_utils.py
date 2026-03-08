"""
Velocity conversion utilities for Vp, Vs, and Poisson's ratio calculations.

Provides functions to convert between seismic velocities and elastic parameters,
with smart suggestions based on typical soil/rock properties.
"""

import math
from typing import Tuple


class VelocityConverter:
    """Utility class for velocity and elastic parameter conversions."""

    @staticmethod
    def vp_from_vs_nu(vs: float, nu: float) -> float:
        """
        Calculate Vp from Vs and Poisson's ratio.

        Parameters
        ----------
        vs : float
            Shear wave velocity (m/s).
        nu : float
            Poisson's ratio (dimensionless, 0 < nu < 0.5).

        Returns
        -------
        float
            Compressional wave velocity Vp (m/s).

        Raises
        ------
        ValueError
            If nu is outside valid range (0, 0.5).
        """
        if nu <= 0 or nu >= 0.5:
            raise ValueError(f"Poisson's ratio must be between 0 and 0.5, got {nu}")
        
        factor = math.sqrt((2 * (1 - nu)) / (1 - 2 * nu))
        return vs * factor

    @staticmethod
    def nu_from_vp_vs(vp: float, vs: float) -> float:
        """
        Calculate Poisson's ratio from Vp and Vs.

        Parameters
        ----------
        vp : float
            Compressional wave velocity (m/s).
        vs : float
            Shear wave velocity (m/s).

        Returns
        -------
        float
            Poisson's ratio (dimensionless).

        Raises
        ------
        ValueError
            If Vp/Vs ratio is invalid (must be > sqrt(2)).
        """
        if vs <= 0:
            raise ValueError(f"Vs must be positive, got {vs}")
        if vp <= 0:
            raise ValueError(f"Vp must be positive, got {vp}")
        
        ratio = vp / vs
        if ratio < math.sqrt(2):
            raise ValueError(
                f"Vp/Vs ratio must be >= sqrt(2) (~1.414), got {ratio:.3f}"
            )
        
        ratio_sq = ratio ** 2
        nu = (ratio_sq - 2) / (2 * (ratio_sq - 1))
        return nu

    @staticmethod
    def vs_from_vp_nu(vp: float, nu: float) -> float:
        """
        Calculate Vs from Vp and Poisson's ratio.

        Parameters
        ----------
        vp : float
            Compressional wave velocity (m/s).
        nu : float
            Poisson's ratio (dimensionless, 0 < nu < 0.5).

        Returns
        -------
        float
            Shear wave velocity Vs (m/s).
        """
        if nu <= 0 or nu >= 0.5:
            raise ValueError(f"Poisson's ratio must be between 0 and 0.5, got {nu}")
        
        factor = math.sqrt((2 * (1 - nu)) / (1 - 2 * nu))
        return vp / factor

    @staticmethod
    def vp_vs_ratio(nu: float) -> float:
        """
        Calculate Vp/Vs ratio from Poisson's ratio.

        Parameters
        ----------
        nu : float
            Poisson's ratio (dimensionless, 0 < nu < 0.5).

        Returns
        -------
        float
            Vp/Vs ratio.
        """
        if nu <= 0 or nu >= 0.5:
            raise ValueError(f"Poisson's ratio must be between 0 and 0.5, got {nu}")
        
        return math.sqrt((2 * (1 - nu)) / (1 - 2 * nu))

    @staticmethod
    def suggest_nu(vs: float) -> float:
        """
        Suggest Poisson's ratio based on Vs value.

        Uses empirical relationships for typical soil/rock types.

        Parameters
        ----------
        vs : float
            Shear wave velocity (m/s).

        Returns
        -------
        float
            Suggested Poisson's ratio.
        """
        if vs < 150:
            return 0.48  # Soft clay, saturated soil
        elif vs < 250:
            return 0.40  # Medium clay, loose sand
        elif vs < 400:
            return 0.33  # Stiff clay, medium sand
        elif vs < 600:
            return 0.28  # Dense sand, gravel
        elif vs < 1000:
            return 0.25  # Weathered rock, very dense soil
        else:
            return 0.22  # Intact rock

    @staticmethod
    def suggest_density(vs: float) -> float:
        """
        Suggest density based on Vs value.

        Uses empirical relationships for typical soil/rock types.

        Parameters
        ----------
        vs : float
            Shear wave velocity (m/s).

        Returns
        -------
        float
            Suggested density (kg/m3).
        """
        if vs < 150:
            return 1700  # Soft clay
        elif vs < 300:
            return 1850  # Medium stiff soil
        elif vs < 500:
            return 2000  # Stiff soil
        elif vs < 1000:
            return 2200  # Very dense soil / weathered rock
        else:
            return 2500  # Rock

    @staticmethod
    def get_soil_type_description(vs: float) -> str:
        """
        Get a description of the soil type based on Vs.

        Parameters
        ----------
        vs : float
            Shear wave velocity (m/s).

        Returns
        -------
        str
            Description of the soil type.
        """
        if vs < 150:
            return "Soft clay / saturated soil"
        elif vs < 250:
            return "Medium clay / loose sand"
        elif vs < 400:
            return "Stiff clay / medium sand"
        elif vs < 600:
            return "Dense sand / gravel"
        elif vs < 1000:
            return "Weathered rock / very dense soil"
        else:
            return "Intact rock"

    @staticmethod
    def get_typical_values_table() -> list:
        """
        Get a table of typical Poisson's ratio values for different materials.

        Returns
        -------
        list
            List of tuples (material, nu_min, nu_max, nu_typical).
        """
        return [
            ("Soft clay (saturated)", 0.45, 0.50, 0.48),
            ("Medium clay", 0.35, 0.45, 0.40),
            ("Stiff clay", 0.30, 0.35, 0.33),
            ("Loose sand", 0.35, 0.40, 0.38),
            ("Medium sand", 0.30, 0.35, 0.32),
            ("Dense sand", 0.25, 0.30, 0.28),
            ("Gravel", 0.25, 0.35, 0.30),
            ("Weathered rock", 0.22, 0.28, 0.25),
            ("Intact rock", 0.18, 0.25, 0.22),
        ]

    @staticmethod
    def validate_velocities(vs: float, vp: float) -> Tuple[bool, str]:
        """
        Validate that Vp and Vs are physically consistent.

        Parameters
        ----------
        vs : float
            Shear wave velocity (m/s).
        vp : float
            Compressional wave velocity (m/s).

        Returns
        -------
        Tuple[bool, str]
            (is_valid, message)
        """
        if vs <= 0:
            return False, "Vs must be positive"
        if vp <= 0:
            return False, "Vp must be positive"
        
        ratio = vp / vs
        min_ratio = math.sqrt(2)  # ~1.414
        
        if ratio < min_ratio:
            return False, f"Vp/Vs ratio ({ratio:.2f}) must be >= {min_ratio:.3f}"
        
        if ratio > 10:
            return False, f"Vp/Vs ratio ({ratio:.2f}) is unusually high (> 10)"
        
        return True, "Valid"
