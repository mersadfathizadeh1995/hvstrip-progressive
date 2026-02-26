"""
Tests for soil_profile module.
"""

import tempfile
from pathlib import Path

import pytest

from hvstrip_progressive.core.soil_profile import Layer, SoilProfile
from hvstrip_progressive.core.velocity_utils import VelocityConverter


class TestLayer:
    """Tests for Layer dataclass."""

    def test_layer_creation(self):
        """Test basic layer creation."""
        layer = Layer(thickness=5.0, vs=300.0)
        assert layer.thickness == 5.0
        assert layer.vs == 300.0
        assert layer.vp is None
        assert layer.nu is None
        assert layer.density == 2000.0
        assert layer.is_halfspace is False

    def test_layer_with_vp(self):
        """Test layer with explicit Vp."""
        layer = Layer(thickness=5.0, vs=300.0, vp=600.0)
        assert layer.compute_vp() == 600.0

    def test_layer_compute_vp_from_nu(self):
        """Test Vp computation from nu."""
        layer = Layer(thickness=5.0, vs=300.0, nu=0.30)
        expected_vp = VelocityConverter.vp_from_vs_nu(300.0, 0.30)
        assert abs(layer.compute_vp() - expected_vp) < 0.01

    def test_layer_compute_vp_auto(self):
        """Test automatic Vp computation using suggested nu."""
        layer = Layer(thickness=5.0, vs=300.0)
        suggested_nu = VelocityConverter.suggest_nu(300.0)
        expected_vp = VelocityConverter.vp_from_vs_nu(300.0, suggested_nu)
        assert abs(layer.compute_vp() - expected_vp) < 0.01

    def test_layer_get_effective_nu_explicit(self):
        """Test effective nu with explicit nu."""
        layer = Layer(thickness=5.0, vs=300.0, nu=0.35)
        assert layer.get_effective_nu() == 0.35

    def test_layer_get_effective_nu_from_vp(self):
        """Test effective nu computed from Vp."""
        layer = Layer(thickness=5.0, vs=300.0, vp=600.0)
        expected_nu = VelocityConverter.nu_from_vp_vs(600.0, 300.0)
        assert abs(layer.get_effective_nu() - expected_nu) < 0.001

    def test_layer_get_soil_type(self):
        """Test soil type description."""
        layer = Layer(thickness=5.0, vs=100.0)
        assert "soft" in layer.get_soil_type().lower()

    def test_layer_validate_valid(self):
        """Test validation of valid layer."""
        layer = Layer(thickness=5.0, vs=300.0, density=2000.0)
        valid, msg = layer.validate()
        assert valid

    def test_layer_validate_invalid_vs(self):
        """Test validation fails for invalid Vs."""
        layer = Layer(thickness=5.0, vs=-100.0)
        valid, msg = layer.validate()
        assert not valid
        assert "Vs" in msg

    def test_layer_validate_invalid_thickness(self):
        """Test validation fails for invalid thickness."""
        layer = Layer(thickness=-5.0, vs=300.0)
        valid, msg = layer.validate()
        assert not valid

    def test_layer_halfspace(self):
        """Test half-space layer."""
        layer = Layer(thickness=0, vs=800.0, is_halfspace=True)
        valid, msg = layer.validate()
        assert valid


class TestSoilProfile:
    """Tests for SoilProfile dataclass."""

    def test_profile_creation(self):
        """Test basic profile creation."""
        profile = SoilProfile(name="Test")
        assert profile.name == "Test"
        assert len(profile.layers) == 0

    def test_add_layer(self):
        """Test adding layers."""
        profile = SoilProfile()
        layer = Layer(thickness=5.0, vs=300.0)
        profile.add_layer(layer)
        assert len(profile.layers) == 1

    def test_remove_layer(self):
        """Test removing layers."""
        profile = SoilProfile()
        profile.add_layer(Layer(thickness=5.0, vs=300.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, is_halfspace=True))
        profile.remove_layer(0)
        assert len(profile.layers) == 1

    def test_move_layer(self):
        """Test moving layers."""
        profile = SoilProfile()
        profile.add_layer(Layer(thickness=5.0, vs=200.0))
        profile.add_layer(Layer(thickness=10.0, vs=400.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, is_halfspace=True))
        
        profile.move_layer(0, 1)
        assert profile.layers[0].vs == 400.0
        assert profile.layers[1].vs == 200.0

    def test_get_total_thickness(self):
        """Test total thickness calculation."""
        profile = SoilProfile()
        profile.add_layer(Layer(thickness=5.0, vs=200.0))
        profile.add_layer(Layer(thickness=10.0, vs=400.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, is_halfspace=True))
        
        assert profile.get_total_thickness() == 15.0

    def test_get_depth_to_layer(self):
        """Test depth calculation."""
        profile = SoilProfile()
        profile.add_layer(Layer(thickness=5.0, vs=200.0))
        profile.add_layer(Layer(thickness=10.0, vs=400.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, is_halfspace=True))
        
        assert profile.get_depth_to_layer(0) == 0.0
        assert profile.get_depth_to_layer(1) == 5.0
        assert profile.get_depth_to_layer(2) == 15.0

    def test_validate_valid_profile(self):
        """Test validation of valid profile."""
        profile = SoilProfile()
        profile.add_layer(Layer(thickness=5.0, vs=200.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, is_halfspace=True))
        
        valid, errors = profile.validate()
        assert valid
        assert len(errors) == 0

    def test_validate_empty_profile(self):
        """Test validation fails for empty profile."""
        profile = SoilProfile()
        valid, errors = profile.validate()
        assert not valid
        assert any("at least one" in e.lower() for e in errors)

    def test_validate_no_halfspace(self):
        """Test validation fails without half-space."""
        profile = SoilProfile()
        profile.add_layer(Layer(thickness=5.0, vs=200.0))
        
        valid, errors = profile.validate()
        assert not valid
        assert any("half-space" in e.lower() for e in errors)

    def test_to_hvf_format(self):
        """Test HVf format export."""
        profile = SoilProfile()
        profile.add_layer(Layer(thickness=5.0, vs=200.0, vp=400.0, density=1800.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, vp=1600.0, density=2200.0, is_halfspace=True))
        
        hvf = profile.to_hvf_format()
        lines = hvf.strip().split("\n")
        
        assert lines[0] == "2"
        assert "5.00" in lines[1]
        assert "200.0" in lines[1]

    def test_to_csv(self):
        """Test CSV export."""
        profile = SoilProfile()
        profile.add_layer(Layer(thickness=5.0, vs=200.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, is_halfspace=True))
        
        csv = profile.to_csv()
        lines = csv.strip().split("\n")
        
        assert "thickness" in lines[0].lower()
        assert "vs" in lines[0].lower()
        assert len(lines) == 3

    def test_from_hvf_string(self):
        """Test HVf format parsing."""
        hvf_content = """2
5.00 400.0 200.0 1.800
0.00 1600.0 800.0 2.200"""
        
        profile = SoilProfile.from_hvf_string(hvf_content, name="Test")
        
        assert profile.name == "Test"
        assert len(profile.layers) == 2
        assert profile.layers[0].thickness == 5.0
        assert profile.layers[0].vs == 200.0
        assert profile.layers[0].vp == 400.0
        assert profile.layers[1].is_halfspace

    def test_save_and_load_hvf(self):
        """Test HVf file save and load roundtrip."""
        profile = SoilProfile(name="Test")
        profile.add_layer(Layer(thickness=5.0, vs=200.0, vp=400.0, density=1800.0))
        profile.add_layer(Layer(thickness=10.0, vs=400.0, vp=800.0, density=2000.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, vp=1600.0, density=2200.0, is_halfspace=True))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_profile.txt"
            profile.save_hvf(str(filepath))
            
            loaded = SoilProfile.from_hvf_file(str(filepath))
            
            assert len(loaded.layers) == 3
            assert loaded.layers[0].vs == 200.0
            assert loaded.layers[2].is_halfspace

    def test_save_and_load_csv(self):
        """Test CSV file save and load roundtrip."""
        profile = SoilProfile(name="Test")
        profile.add_layer(Layer(thickness=5.0, vs=200.0, nu=0.40, density=1800.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, nu=0.25, density=2200.0, is_halfspace=True))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_profile.csv"
            profile.save_csv(str(filepath), include_computed=False)
            
            loaded = SoilProfile.from_csv_file(str(filepath))
            
            assert len(loaded.layers) == 2
            assert loaded.layers[0].vs == 200.0

    def test_copy(self):
        """Test profile copy."""
        profile = SoilProfile(name="Original")
        profile.add_layer(Layer(thickness=5.0, vs=200.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, is_halfspace=True))
        
        copy = profile.copy()
        
        assert copy.name == profile.name
        assert len(copy.layers) == len(profile.layers)
        
        copy.layers[0].vs = 300.0
        assert profile.layers[0].vs == 200.0


class TestFromAuto:
    """Tests for SoilProfile.from_auto format detection."""

    def test_auto_hvf_txt(self, tmp_path):
        """from_auto loads HVf-format .txt files."""
        hvf = "2\n5.0  400.0  200.0  1800.0\n0.0  1600.0  800.0  2200.0\n"
        f = tmp_path / "model.txt"
        f.write_text(hvf)
        p = SoilProfile.from_auto(str(f))
        assert len(p.layers) == 2
        assert p.layers[0].vs == 200.0

    def test_auto_csv(self, tmp_path):
        """from_auto loads CSV files."""
        csv_text = (
            "thickness,vs,vp,density\n"
            "5.0,200.0,400.0,1800.0\n"
            "0.0,800.0,1600.0,2200.0\n"
        )
        f = tmp_path / "profile.csv"
        f.write_text(csv_text)
        p = SoilProfile.from_auto(str(f))
        assert len(p.layers) == 2
        assert p.layers[1].vs == 800.0

    def test_auto_name_override(self, tmp_path):
        """from_auto respects explicit name parameter."""
        hvf = "2\n5.0  400.0  200.0  1800.0\n0.0  1600.0  800.0  2200.0\n"
        f = tmp_path / "model.txt"
        f.write_text(hvf)
        p = SoilProfile.from_auto(str(f), name="custom_name")
        assert p.name == "custom_name"

    def test_auto_defaults_name_to_stem(self, tmp_path):
        """from_auto uses file stem as name when not provided."""
        hvf = "2\n5.0  400.0  200.0  1800.0\n0.0  1600.0  800.0  2200.0\n"
        f = tmp_path / "my_profile.txt"
        f.write_text(hvf)
        p = SoilProfile.from_auto(str(f))
        assert p.name == "my_profile"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
