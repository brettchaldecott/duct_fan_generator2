"""Tests for configuration loading and validation."""

import pytest
from copy import deepcopy
from src.config import load_config, validate_config, compute_derived, ConfigError


class TestConfigLoading:
    """Test that valid configs load correctly."""

    def test_default_config_loads(self, default_config):
        """Valid default config loads without error."""
        assert default_config is not None
        assert "motor" in default_config
        assert "gears" in default_config
        assert "derived" in default_config

    def test_config_has_all_sections(self, default_config):
        """Config contains all required top-level sections."""
        required = [
            "motor", "duct", "gears", "magnetic_coupling",
            "blades", "hub", "stators", "bearings", "materials", "print"
        ]
        for section in required:
            assert section in default_config, f"Missing config section: {section}"

    def test_derived_values_present(self, default_config):
        """Derived values are computed and present."""
        derived = default_config["derived"]
        assert "sun_pitch_diameter" in derived
        assert "gear_ratio" in derived
        assert "stage_rpms" in derived
        assert "hub_od" in derived
        assert "blade_span" in derived


class TestGearConstraints:
    """Test gear tooth constraints."""

    def test_ring_equals_sun_plus_2_planet(self, default_config):
        """Ring teeth = sun teeth + 2 * planet teeth."""
        g = default_config["gears"]
        assert g["ring_teeth"] == g["sun_teeth"] + 2 * g["planet_teeth"]

    def test_invalid_gear_ratio_raises(self, default_config):
        """Invalid gear tooth relationship raises ConfigError."""
        bad = deepcopy(default_config)
        del bad["derived"]  # remove derived so validate doesn't get confused
        bad["gears"]["ring_teeth"] = 35  # wrong: should be 20 + 2*10 = 40
        with pytest.raises(ConfigError, match="Gear tooth constraint"):
            validate_config(bad)


class TestInvalidConfigs:
    """Test that invalid configurations raise clear errors."""

    def test_negative_duct_diameter(self, default_config):
        """Negative duct diameter raises ConfigError."""
        bad = deepcopy(default_config)
        del bad["derived"]
        bad["duct"]["inner_diameter"] = -10
        with pytest.raises(ConfigError, match="Duct inner_diameter must be positive"):
            validate_config(bad)

    def test_negative_hub_wall(self, default_config):
        """Negative hub wall thickness raises ConfigError."""
        bad = deepcopy(default_config)
        del bad["derived"]
        bad["hub"]["wall_thickness"] = -1
        with pytest.raises(ConfigError, match="Hub wall_thickness must be positive"):
            validate_config(bad)

    def test_zero_gear_module(self, default_config):
        """Zero gear module raises ConfigError."""
        bad = deepcopy(default_config)
        del bad["derived"]
        bad["gears"]["module"] = 0
        with pytest.raises(ConfigError, match="Gear module must be positive"):
            validate_config(bad)

    def test_mismatched_stages(self, default_config):
        """Mismatched blade/coupling stage count raises ConfigError."""
        bad = deepcopy(default_config)
        del bad["derived"]
        bad["blades"]["stages"] = bad["blades"]["stages"][:2]  # only 2 blade stages
        with pytest.raises(ConfigError, match="Number of blade stages"):
            validate_config(bad)


class TestDerivedValues:
    """Test that derived values are computed correctly."""

    def test_gear_ratio(self, default_config):
        """Gear ratio = ring_teeth / sun_teeth = 40/20 = 2.0."""
        assert default_config["derived"]["gear_ratio"] == 2.0

    def test_stage_rpms(self, default_config):
        """Stage RPMs follow gear reduction chain."""
        rpms = default_config["derived"]["stage_rpms"]
        # Stage 1: 12000 RPM (motor direct)
        assert rpms[0] == pytest.approx(12000.0)
        # Stage 2: 12000 / 2.0 = 6000 RPM
        assert rpms[1] == pytest.approx(6000.0)
        # Stage 3: 6000 / 2.0 = 3000 RPM
        assert rpms[2] == pytest.approx(3000.0)

    def test_rotation_directions_alternate(self, default_config):
        """Rotation directions alternate with each gear stage."""
        dirs = default_config["derived"]["stage_directions"]
        assert dirs[0] == 1   # CW
        assert dirs[1] == -1  # CCW (reversed by first gear stage)
        assert dirs[2] == 1   # CW (reversed again)

    def test_pitch_diameters(self, default_config):
        """Pitch diameters = module * teeth."""
        d = default_config["derived"]
        g = default_config["gears"]
        assert d["sun_pitch_diameter"] == g["module"] * g["sun_teeth"]
        assert d["planet_pitch_diameter"] == g["module"] * g["planet_teeth"]
        assert d["ring_pitch_diameter"] == g["module"] * g["ring_teeth"]

    def test_blade_span_positive(self, default_config):
        """Blade span is positive and reasonable."""
        span = default_config["derived"]["blade_span"]
        assert span > 0
        # Should be roughly (135 - tip_clearance) - hub_radius
        assert span > 30  # at least 30mm span
        assert span < 150  # less than duct radius

    def test_hub_od_larger_than_motor(self, default_config):
        """Hub OD must be larger than motor body diameter."""
        assert (
            default_config["derived"]["hub_od"]
            > default_config["motor"]["body_diameter"]
        )
