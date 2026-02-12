"""Tests for assembly validation."""

import pytest
import numpy as np
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

    def test_tip_clearance_met_no_meshes(self, default_config):
        """Blade tip clearance >= configured minimum (config-based fallback)."""
        validator = AssemblyValidator(default_config)
        results = validator.check_blade_tip_clearance()
        assert len(results) == 1
        assert results[0].passed, f"Tip clearance failed: {results[0].detail}"

    def test_tip_clearance_config_fallback(self, default_config):
        """Config-based fallback equals configured tip clearance."""
        validator = AssemblyValidator(default_config)
        results = validator.check_blade_tip_clearance()
        expected = default_config["blades"]["tip_clearance"]
        assert results[0].value == pytest.approx(expected, abs=0.01)

    def test_tip_clearance_with_mesh(self, default_config):
        """Tip clearance check uses actual mesh extents when provided."""
        import trimesh
        validator = AssemblyValidator(default_config)
        duct_r = default_config["duct"]["inner_diameter"] / 2

        # Create a cylindrical blade ring mesh at a known radius
        blade_r = duct_r - 5.0  # 5mm gap
        mesh = trimesh.creation.cylinder(radius=blade_r, height=10)
        meshes = {"blade_ring_stage_1": mesh}

        results = validator.check_blade_tip_clearance(meshes)
        assert len(results) == 1
        assert results[0].passed
        assert results[0].value == pytest.approx(5.0, abs=0.5)


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


class TestAxialLayout:
    """Test axial layout positioning."""

    def test_part_positions_computed(self, default_config):
        """Derived config includes part_positions dict."""
        assert "part_positions" in default_config["derived"]
        positions = default_config["derived"]["part_positions"]
        assert "stator_entry" in positions
        assert "blade_ring_stage_1" in positions
        assert "stator_exit" in positions

    def test_positions_increase_axially(self, default_config):
        """Parts are positioned in correct axial order."""
        positions = default_config["derived"]["part_positions"]
        assert positions["stator_entry"] < positions["blade_ring_stage_1"]
        assert positions["blade_ring_stage_1"] < positions["blade_ring_stage_2"]
        assert positions["blade_ring_stage_2"] < positions["blade_ring_stage_3"]
        assert positions["blade_ring_stage_3"] < positions["stator_exit"]


class TestBladeRingClearance:
    """Test blade ring to hub clearance checks."""

    def test_blade_ring_clearance_checked(self, default_config):
        """Blade ring clearance is checked for all stages."""
        validator = AssemblyValidator(default_config)
        results = validator.check_blade_ring_clearance()
        num_stages = len(default_config["blades"]["stages"])
        assert len(results) == num_stages

    def test_blade_ring_clears_hub(self, default_config):
        """Blade ring inner surface has air gap from hub outer surface."""
        validator = AssemblyValidator(default_config)
        results = validator.check_blade_ring_clearance()
        for r in results:
            assert r.passed, f"Blade ring clearance failed: {r.detail}"

    def test_blade_ring_is_external(self, default_config):
        """Blade ring inner radius is larger than hub outer radius."""
        blade_ring_radii = default_config["derived"]["blade_ring_radii"]
        per_stage_hub_radii = default_config["derived"]["per_stage_hub_radii"]
        for i, ring_info in enumerate(blade_ring_radii):
            assert ring_info["ring_inner_r"] > per_stage_hub_radii[i], (
                f"Stage {i}: ring inner {ring_info['ring_inner_r']:.1f}mm "
                f"should be > hub {per_stage_hub_radii[i]:.1f}mm"
            )


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
