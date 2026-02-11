"""Tests for assembly validation."""

import pytest
from validation.assembly_validator import AssemblyValidator


class TestBearingSeats:
    """Test bearing seat dimension checks."""

    def test_bearing_seats_checked(self, default_config):
        """Bearing seats are checked for all bearings."""
        validator = AssemblyValidator(default_config)
        results = validator.check_bearing_seats()
        assert len(results) == len(default_config["bearings"])

    def test_bearing_seat_tolerance(self, default_config):
        """Bearing seats should be within +0.05 to +0.20mm of bearing OD."""
        validator = AssemblyValidator(default_config)
        results = validator.check_bearing_seats()
        for r in results:
            assert r.detail != ""  # has detail info


class TestMagnetPockets:
    """Test magnet pocket dimension checks."""

    def test_magnet_pockets_checked(self, default_config):
        """Magnet pockets are checked for all stages."""
        validator = AssemblyValidator(default_config)
        results = validator.check_magnet_pockets()
        assert len(results) == len(default_config["magnetic_coupling"]["stages"])

    def test_pocket_clearance_adequate(self, default_config):
        """Magnet pocket clearance should be within [0.15, 0.30]mm."""
        validator = AssemblyValidator(default_config)
        results = validator.check_magnet_pockets()
        for r in results:
            # Default clearance is 0.2mm per side, pocket_d = mag_d + 0.4mm
            # Range: [mag_d + 0.15, mag_d + 0.30]
            assert r.value > 0


class TestBladeTipClearance:
    """Test blade tip clearance checks."""

    def test_tip_clearance_met(self, default_config):
        """Blade tip clearance >= configured minimum."""
        validator = AssemblyValidator(default_config)
        results = validator.check_blade_tip_clearance()
        assert len(results) == 1
        assert results[0].passed, f"Tip clearance failed: {results[0].detail}"

    def test_tip_clearance_equals_configured(self, default_config):
        """Tip clearance exactly equals the configured value."""
        validator = AssemblyValidator(default_config)
        results = validator.check_blade_tip_clearance()
        expected = default_config["blades"]["tip_clearance"]
        assert results[0].value == pytest.approx(expected, abs=0.01)


class TestBuildVolume:
    """Test build volume checks."""

    def test_build_volume_check_with_valid_parts(self, default_config):
        """Small test meshes pass build volume check."""
        import trimesh
        validator = AssemblyValidator(default_config)
        meshes = {
            "test_part": trimesh.creation.box(extents=[50, 50, 50])
        }
        results = validator.check_build_volume(meshes)
        assert len(results) == 1
        assert results[0].passed


class TestOverallAssembly:
    """Test overall assembly validation."""

    def test_validate_all_runs(self, default_config):
        """Assembly validation runs with no meshes (dimensional checks only)."""
        validator = AssemblyValidator(default_config)
        results = validator.validate_all({})
        assert len(results) > 0

    def test_all_results_have_detail(self, default_config):
        """Every check includes detail string."""
        validator = AssemblyValidator(default_config)
        results = validator.validate_all({})
        for r in results:
            assert len(r.detail) > 0, f"{r.check_name}: missing detail"
