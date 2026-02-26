"""Tests for the engine abstraction layer."""

import pytest
from hvstrip_progressive.core.engines import (
    registry,
    BaseForwardEngine,
    EngineResult,
    DiffuseFieldEngine,
    SHWaveEngine,
    EllipticityEngine,
)


class TestEngineRegistry:
    """Test engine registration and lookup."""

    def test_registry_has_three_engines(self):
        available = registry.list_available()
        assert "diffuse_field" in available
        assert "sh_wave" in available
        assert "ellipticity" in available

    def test_get_diffuse_field(self):
        engine = registry.get("diffuse_field")
        assert isinstance(engine, DiffuseFieldEngine)
        assert engine.name == "diffuse_field"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown engine"):
            registry.get("nonexistent_engine")

    def test_engine_info_returns_list(self):
        info = registry.get_engine_info()
        assert isinstance(info, list)
        assert len(info) >= 3
        names = [i["name"] for i in info]
        assert "diffuse_field" in names


class TestDiffuseFieldEngine:
    """Test the diffuse field engine interface."""

    def test_default_config_has_required_keys(self):
        engine = DiffuseFieldEngine()
        cfg = engine.get_default_config()
        assert "exe_path" in cfg
        assert "fmin" in cfg
        assert "fmax" in cfg
        assert "nf" in cfg

    def test_format_model_returns_hvf_string(self):
        from hvstrip_progressive.core.soil_profile import SoilProfile, Layer

        engine = DiffuseFieldEngine()
        profile = SoilProfile(name="test")
        profile.add_layer(Layer(thickness=5.0, vs=200.0, vp=400.0, density=1800.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, vp=1600.0, density=2200.0, is_halfspace=True))

        model_text = engine.format_model(profile)
        lines = model_text.strip().splitlines()
        assert lines[0].strip() == "2"
        assert len(lines) == 3

    def test_repr(self):
        engine = DiffuseFieldEngine()
        assert "DiffuseFieldEngine" in repr(engine)


class TestStubEngines:
    """Test that stub engines raise NotImplementedError."""

    def test_sh_wave_compute_missing_file_raises(self):
        engine = SHWaveEngine()
        with pytest.raises(FileNotFoundError):
            engine.compute("dummy.txt", {})

    def test_ellipticity_compute_missing_file_raises(self):
        engine = EllipticityEngine()
        with pytest.raises(FileNotFoundError):
            engine.compute("dummy.txt", {})

    def test_sh_wave_default_config(self):
        engine = SHWaveEngine()
        cfg = engine.get_default_config()
        assert "fmin" in cfg
        assert "n_samples" in cfg
        assert "Drock" in cfg
        assert "Dsoil" in cfg

    def test_ellipticity_default_config(self):
        engine = EllipticityEngine()
        cfg = engine.get_default_config()
        assert "fmin" in cfg
        assert "gpell_path" in cfg
        assert "n_samples" in cfg


class TestSHWaveEngineIntegration:
    """Integration tests for the SH wave engine with real model files."""

    def test_compute_exampl_model2(self):
        """Verify SH engine produces valid TF from exampl_model2.txt."""
        import os
        model_path = os.path.join("examples", "exampl_model2.txt")
        if not os.path.exists(model_path):
            pytest.skip("exampl_model2.txt not found")

        engine = SHWaveEngine()
        result = engine.compute(model_path)

        assert result.frequencies.shape == (512,)
        assert result.amplitudes.shape == (512,)
        assert result.amplitudes.max() > 1.0  # TF must have amplification
        f0 = result.metadata["f0_hz"][0]
        assert 5.0 < f0 < 15.0  # reasonable F0 range for this model

    def test_compute_with_custom_config(self):
        """Verify custom config overrides work."""
        import os
        model_path = os.path.join("examples", "exampl_model2.txt")
        if not os.path.exists(model_path):
            pytest.skip("exampl_model2.txt not found")

        engine = SHWaveEngine()
        result = engine.compute(model_path, {
            "fmin": 1.0,
            "fmax": 20.0,
            "n_samples": 256,
            "sampling": "linear",
        })

        assert result.frequencies.shape == (256,)
        assert result.amplitudes.shape == (256,)
        assert abs(result.frequencies[0] - 1.0) < 0.1
        assert abs(result.frequencies[-1] - 20.0) < 0.1

    def test_format_model_returns_hvf_string(self):
        """Verify format_model produces valid HVf text."""
        from hvstrip_progressive.core.soil_profile import SoilProfile, Layer

        engine = SHWaveEngine()
        profile = SoilProfile(name="test")
        profile.add_layer(Layer(thickness=5.0, vs=200.0, vp=400.0, density=1800.0))
        profile.add_layer(Layer(thickness=0, vs=800.0, vp=1600.0, density=2200.0, is_halfspace=True))

        model_text = engine.format_model(profile)
        lines = model_text.strip().splitlines()
        assert lines[0].strip() == "2"
        assert len(lines) == 3

    def test_validate_config_rejects_bad_values(self):
        engine = SHWaveEngine()
        ok, msg = engine.validate_config({"fmin": 10, "fmax": 5})
        assert not ok
        assert "fmin" in msg

    def test_registry_lists_sh_wave_as_implemented(self):
        implemented = registry.list_implemented()
        assert "sh_wave" in implemented


class TestHvForwardFacade:
    """Test the backward-compatible hv_forward facade."""

    def test_default_config_matches_engine(self):
        from hvstrip_progressive.core.hv_forward import DEFAULT_CONFIG

        engine_cfg = DiffuseFieldEngine().get_default_config()
        assert DEFAULT_CONFIG["fmin"] == engine_cfg["fmin"]
        assert DEFAULT_CONFIG["fmax"] == engine_cfg["fmax"]
        assert DEFAULT_CONFIG["nf"] == engine_cfg["nf"]
