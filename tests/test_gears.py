"""Tests for gear generation and validation."""

import math
import pytest
from src.gear_generator import GearGenerator
from validation.gear_validator import GearValidator


class TestGearSpecs:
    """Test gear specification computation."""

    def test_gear_specs_computed(self, default_config):
        """All gear specs (sun, planet, ring) are computed."""
        gen = GearGenerator(default_config)
        specs = gen.compute_gear_specs()
        assert "sun" in specs
        assert "planet" in specs
        assert "ring" in specs

    def test_pitch_diameters(self, default_config):
        """Pitch diameters match module × teeth."""
        gen = GearGenerator(default_config)
        specs = gen.compute_gear_specs()
        m = default_config["gears"]["module"]
        assert specs["sun"].pitch_diameter == m * default_config["gears"]["sun_teeth"]
        assert specs["planet"].pitch_diameter == m * default_config["gears"]["planet_teeth"]
        assert specs["ring"].pitch_diameter == m * default_config["gears"]["ring_teeth"]

    def test_ring_is_internal(self, default_config):
        """Ring gear is flagged as internal."""
        gen = GearGenerator(default_config)
        specs = gen.compute_gear_specs()
        assert specs["ring"].is_internal is True
        assert specs["sun"].is_internal is False


class TestContactRatio:
    """Test contact ratio calculations."""

    def test_sun_planet_contact_ratio(self, default_config):
        """Sun-planet contact ratio >= 1.2."""
        gen = GearGenerator(default_config)
        specs = gen.compute_gear_specs()
        cr = gen.contact_ratio(specs["sun"], specs["planet"])
        assert cr >= 1.2, f"Sun-planet CR={cr:.3f} < 1.2"

    def test_planet_ring_contact_ratio(self, default_config):
        """Planet-ring contact ratio should be positive and >= 1.2."""
        gen = GearGenerator(default_config)
        specs = gen.compute_gear_specs()
        cr_pr = gen.contact_ratio(specs["planet"], specs["ring"])
        assert cr_pr >= 1.2, f"Planet-ring CR={cr_pr:.3f} < 1.2"


class TestUndercutting:
    """Test undercutting detection."""

    def test_sun_no_undercut(self, default_config):
        """Sun gear (20 teeth, 20° PA) should NOT be flagged for undercutting."""
        gen = GearGenerator(default_config)
        specs = gen.compute_gear_specs()
        assert not gen.check_undercutting(specs["sun"])

    def test_planet_undercut_risk(self, default_config):
        """Planet gear (10 teeth, 20° PA) SHOULD be flagged for undercutting.

        Minimum teeth for 20° PA is 17.
        """
        gen = GearGenerator(default_config)
        specs = gen.compute_gear_specs()
        assert gen.check_undercutting(specs["planet"]), (
            "10-tooth planet at 20° PA should be flagged for undercutting"
        )

    def test_min_teeth_20deg(self, default_config):
        """Minimum teeth for 20° PA: ceil(2/sin²(20°)) = 18."""
        gen = GearGenerator(default_config)
        # 2 / sin²(20°) = 2 / 0.1170 = 17.097, ceil = 18
        assert gen.min_teeth_no_undercut() == 18


class TestGearRatio:
    """Test gear ratio computation."""

    def test_gear_ratio_is_2(self, default_config):
        """Carrier-fixed ratio = R/S = 40/20 = 2.0."""
        gen = GearGenerator(default_config)
        assert gen.gear_ratio() == 2.0

    def test_planet_positions(self, default_config):
        """3 planets evenly spaced at 120°."""
        gen = GearGenerator(default_config)
        positions = gen.planet_positions()
        assert len(positions) == 3
        for i in range(len(positions)):
            expected = 2 * math.pi * i / 3
            assert abs(positions[i] - expected) < 1e-10


class TestGearValidator:
    """Test gear validation framework."""

    def test_validator_runs(self, default_config):
        """Gear validator runs without error."""
        validator = GearValidator(default_config)
        results = validator.validate_all()
        assert len(results) > 0

    def test_backlash_in_range(self, default_config):
        """Default backlash (0.15mm) is within FDM range [0.10, 0.20]."""
        validator = GearValidator(default_config)
        results = validator.validate_all()
        backlash_result = next(r for r in results if r.check_name == "backlash")
        assert backlash_result.passed

    def test_undercutting_flagged_by_validator(self, default_config):
        """Validator flags planet gear undercutting."""
        validator = GearValidator(default_config)
        results = validator.validate_all()
        planet_undercut = next(
            r for r in results
            if r.check_name == "undercutting" and r.gear_name == "planet"
        )
        assert not planet_undercut.passed  # Should be flagged

    def test_all_results_have_details(self, default_config):
        """All validation results include detail strings."""
        validator = GearValidator(default_config)
        results = validator.validate_all()
        for r in results:
            assert len(r.detail) > 0


class TestGearProfile:
    """Test gear profile generation."""

    def test_involute_profile_generates(self, default_config):
        """Involute profile generates without error."""
        gen = GearGenerator(default_config)
        specs = gen.compute_gear_specs()
        profile = gen.generate_involute_profile(specs["sun"])
        assert profile.shape[0] > 0
        assert profile.shape[1] == 2

    def test_full_gear_profile(self, default_config):
        """Full gear profile with all teeth generates."""
        gen = GearGenerator(default_config)
        specs = gen.compute_gear_specs()
        profile = gen.generate_full_gear_profile(specs["sun"])
        assert profile.shape[0] > specs["sun"].teeth  # at least one point per tooth
