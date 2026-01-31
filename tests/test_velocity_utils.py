"""
Tests for velocity_utils module.
"""

import math
import pytest

from hvstrip_progressive.core.velocity_utils import VelocityConverter


class TestVelocityConverter:
    """Tests for VelocityConverter class."""

    def test_vp_from_vs_nu_basic(self):
        """Test basic Vp calculation from Vs and nu."""
        vs = 300.0
        nu = 0.25
        
        expected_factor = math.sqrt((2 * (1 - nu)) / (1 - 2 * nu))
        expected_vp = vs * expected_factor
        
        result = VelocityConverter.vp_from_vs_nu(vs, nu)
        assert abs(result - expected_vp) < 0.01

    def test_vp_from_vs_nu_soft_clay(self):
        """Test Vp calculation for soft clay (high nu)."""
        vs = 100.0
        nu = 0.48
        
        result = VelocityConverter.vp_from_vs_nu(vs, nu)
        assert result > vs * 4

    def test_vp_from_vs_nu_rock(self):
        """Test Vp calculation for rock (low nu)."""
        vs = 1000.0
        nu = 0.22
        
        result = VelocityConverter.vp_from_vs_nu(vs, nu)
        assert result > vs * 1.4
        assert result < vs * 2.5

    def test_vp_from_vs_nu_invalid_nu(self):
        """Test that invalid nu raises ValueError."""
        vs = 300.0
        
        with pytest.raises(ValueError):
            VelocityConverter.vp_from_vs_nu(vs, 0.0)
        
        with pytest.raises(ValueError):
            VelocityConverter.vp_from_vs_nu(vs, 0.5)
        
        with pytest.raises(ValueError):
            VelocityConverter.vp_from_vs_nu(vs, -0.1)
        
        with pytest.raises(ValueError):
            VelocityConverter.vp_from_vs_nu(vs, 0.6)

    def test_nu_from_vp_vs_basic(self):
        """Test nu calculation from Vp and Vs."""
        vs = 300.0
        nu_original = 0.30
        vp = VelocityConverter.vp_from_vs_nu(vs, nu_original)
        
        result = VelocityConverter.nu_from_vp_vs(vp, vs)
        assert abs(result - nu_original) < 0.001

    def test_nu_from_vp_vs_invalid_ratio(self):
        """Test that invalid Vp/Vs ratio raises ValueError."""
        vs = 300.0
        vp = 300.0
        
        with pytest.raises(ValueError):
            VelocityConverter.nu_from_vp_vs(vp, vs)

    def test_vs_from_vp_nu_roundtrip(self):
        """Test Vs calculation roundtrip."""
        vs_original = 350.0
        nu = 0.28
        vp = VelocityConverter.vp_from_vs_nu(vs_original, nu)
        
        result = VelocityConverter.vs_from_vp_nu(vp, nu)
        assert abs(result - vs_original) < 0.01

    def test_vp_vs_ratio(self):
        """Test Vp/Vs ratio calculation."""
        nu = 0.25
        ratio = VelocityConverter.vp_vs_ratio(nu)
        
        assert ratio > math.sqrt(2)
        
        vs = 300.0
        vp = VelocityConverter.vp_from_vs_nu(vs, nu)
        expected_ratio = vp / vs
        
        assert abs(ratio - expected_ratio) < 0.001

    def test_suggest_nu_soft_clay(self):
        """Test nu suggestion for soft clay."""
        vs = 100.0
        nu = VelocityConverter.suggest_nu(vs)
        assert nu == 0.48

    def test_suggest_nu_medium_clay(self):
        """Test nu suggestion for medium clay."""
        vs = 200.0
        nu = VelocityConverter.suggest_nu(vs)
        assert nu == 0.40

    def test_suggest_nu_stiff_clay(self):
        """Test nu suggestion for stiff clay."""
        vs = 350.0
        nu = VelocityConverter.suggest_nu(vs)
        assert nu == 0.33

    def test_suggest_nu_dense_sand(self):
        """Test nu suggestion for dense sand."""
        vs = 500.0
        nu = VelocityConverter.suggest_nu(vs)
        assert nu == 0.28

    def test_suggest_nu_weathered_rock(self):
        """Test nu suggestion for weathered rock."""
        vs = 800.0
        nu = VelocityConverter.suggest_nu(vs)
        assert nu == 0.25

    def test_suggest_nu_rock(self):
        """Test nu suggestion for rock."""
        vs = 1500.0
        nu = VelocityConverter.suggest_nu(vs)
        assert nu == 0.22

    def test_suggest_density(self):
        """Test density suggestions."""
        assert VelocityConverter.suggest_density(100) == 1700
        assert VelocityConverter.suggest_density(200) == 1850
        assert VelocityConverter.suggest_density(400) == 2000
        assert VelocityConverter.suggest_density(700) == 2200
        assert VelocityConverter.suggest_density(1500) == 2500

    def test_get_soil_type_description(self):
        """Test soil type descriptions."""
        assert "soft" in VelocityConverter.get_soil_type_description(100).lower()
        assert "rock" in VelocityConverter.get_soil_type_description(1500).lower()

    def test_get_typical_values_table(self):
        """Test typical values table."""
        table = VelocityConverter.get_typical_values_table()
        assert len(table) > 0
        assert all(len(row) == 4 for row in table)

    def test_validate_velocities_valid(self):
        """Test velocity validation for valid values."""
        vs = 300.0
        vp = 600.0
        valid, msg = VelocityConverter.validate_velocities(vs, vp)
        assert valid
        assert msg == "Valid"

    def test_validate_velocities_invalid_ratio(self):
        """Test velocity validation for invalid ratio."""
        vs = 300.0
        vp = 350.0
        valid, msg = VelocityConverter.validate_velocities(vs, vp)
        assert not valid
        assert "ratio" in msg.lower()

    def test_validate_velocities_negative(self):
        """Test velocity validation for negative values."""
        valid, msg = VelocityConverter.validate_velocities(-100, 500)
        assert not valid
        
        valid, msg = VelocityConverter.validate_velocities(100, -500)
        assert not valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
