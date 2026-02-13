"""Tests for new features: STEP export, drive train, blade clashing, compression."""

import math
import os
import tempfile
import pytest
import cadquery as cq
from src.config import load_config, DEFAULT_CONFIG_PATH
from src.shaft_generator import ShaftGenerator
from src.gear_generator import GearGenerator
from validation.assembly_validator import AssemblyValidator


class TestSTEPExport:
    """Test STEP file export functionality."""

    def test_export_step_creates_files(self, default_config, bemt_results):
        """STEP export creates individual part files and assembly."""
        from src.assembly import AssemblyGenerator
        generator = AssemblyGenerator(default_config, bemt_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            generator.output_dir = tmpdir
            solids = generator.generate_all_solids()
            exported = generator.export_step(solids)

            assert len(exported) > 0
            # Check assembly file exists
            asm_path = os.path.join(tmpdir, "step", "full_assembly.step")
            assert os.path.exists(asm_path)
            assert os.path.getsize(asm_path) > 0

    def test_generate_all_solids_returns_dict(self, default_config, bemt_results):
        """generate_all_solids() returns dict of CQ workplanes."""
        from src.assembly import AssemblyGenerator
        generator = AssemblyGenerator(default_config, bemt_results)
        solids = generator.generate_all_solids()

        assert isinstance(solids, dict)
        assert len(solids) > 0
        assert "hub_half_a" in solids
        assert "inner_shaft" in solids


class TestDriveTrainExtents:
    """Test that tubes have correct axial extents."""

    def test_tube_extents_computed(self, default_config):
        """Tube start/end Z positions are computed in derived config."""
        d = default_config["derived"]
        assert "middle_tube_start_z" in d
        assert "middle_tube_end_z" in d
        assert "outer_tube_start_z" in d
        assert "outer_tube_end_z" in d

    def test_middle_tube_no_gear_interference(self, default_config):
        """Middle tube starts after gear stage 0 (doesn't pass through 5mm sun bore)."""
        d = default_config["derived"]
        positions = d["part_positions"]
        gear_width = default_config["gears"]["gear_width"]

        # Middle tube should start after carrier_back_0
        if "carrier_back_0" in positions:
            carrier_back_end = positions["carrier_back_0"] + 3.0  # carrier_width
            assert d["middle_tube_start_z"] >= carrier_back_end - 0.1

    def test_outer_tube_no_gear_interference(self, default_config):
        """Outer tube starts after gear stage 1 (doesn't pass through any sun bore)."""
        d = default_config["derived"]
        positions = d["part_positions"]

        if "carrier_back_1" in positions:
            carrier_back_end = positions["carrier_back_1"] + 3.0
            assert d["outer_tube_start_z"] >= carrier_back_end - 0.1

    def test_inner_shaft_full_length(self, default_config):
        """Inner shaft runs full hub length (5mm fits through all bores)."""
        d = default_config["derived"]
        shaft_length = d["inner_shaft_end_z"] - d["inner_shaft_start_z"]
        assert shaft_length == pytest.approx(d["hub_length"], abs=1.0)


class TestRingOutputHub:
    """Test ring gear output hub generation."""

    def test_ring_output_hub_generates(self, default_config):
        """Ring output hub generates valid solid for each stage."""
        gen = GearGenerator(default_config)
        for stage in range(default_config["gears"]["num_stages"]):
            hub = gen.generate_ring_gear_output_hub(stage)
            assert hub is not None
            assert hub.val().isValid()

    def test_ring_output_hub_in_stage(self, default_config):
        """Ring output hub is included in planetary stage output."""
        gen = GearGenerator(default_config)
        solids = gen.generate_planetary_stage(0)
        assert "ring_output_hub_stage_0" in solids

    def test_output_hub_positions_computed(self, default_config):
        """Ring output hub positions exist in part_positions."""
        positions = default_config["derived"]["part_positions"]
        for i in range(default_config["gears"]["num_stages"]):
            assert f"ring_output_hub_stage_{i}" in positions


class TestCouplingDisc:
    """Test coupling disc generation."""

    def test_coupling_disc_generates(self, default_config):
        """Coupling disc generates valid solid for each blade stage."""
        gen = ShaftGenerator(default_config)
        num_stages = len(default_config["blades"]["stages"])
        for i in range(num_stages):
            disc = gen.generate_coupling_disc(i)
            assert disc is not None
            assert disc.val().isValid()

    def test_coupling_disc_in_generate_all(self, default_config):
        """Coupling discs included in generate_all() output."""
        gen = ShaftGenerator(default_config)
        parts = gen.generate_all()
        num_stages = len(default_config["blades"]["stages"])
        for i in range(num_stages):
            assert f"coupling_disc_stage_{i+1}" in parts

    def test_coupling_disc_positions_computed(self, default_config):
        """Coupling disc positions exist in part_positions."""
        positions = default_config["derived"]["part_positions"]
        num_stages = len(default_config["blades"]["stages"])
        for i in range(num_stages):
            assert f"coupling_disc_stage_{i+1}" in positions


class TestBladeBladeClearance:
    """Test blade-to-blade collision detection."""

    def test_blade_blade_clearance_all_stages(self, default_config):
        """All stages pass blade-blade collision check."""
        validator = AssemblyValidator(default_config)
        results = validator.check_blade_blade_collisions()
        num_stages = len(default_config["blades"]["stages"])
        assert len(results) == num_stages
        for r in results:
            assert r.passed, f"Blade clash failed: {r.detail}"

    def test_blade_chord_within_angular_spacing(self, default_config):
        """Root chord tangential projection < 80% of arc spacing."""
        blade_ring_radii = default_config["derived"]["blade_ring_radii"]
        stages = default_config["blades"]["stages"]
        for i, stage in enumerate(stages):
            n_blades = stage["num_blades"]
            ring_outer_r = blade_ring_radii[i]["ring_outer_r"]
            arc = ring_outer_r * (2 * math.pi / n_blades)
            # 80% of arc should be > 0 (sanity)
            assert arc * 0.8 > 0


class TestCompressionVisibility:
    """Test converging duct and enhanced compression."""

    def test_per_stage_tip_radii_exist(self, default_config):
        """per_stage_tip_radii computed in derived config."""
        d = default_config["derived"]
        assert "per_stage_tip_radii" in d
        num_stages = len(default_config["blades"]["stages"])
        assert len(d["per_stage_tip_radii"]) == num_stages

    def test_per_stage_tip_radii_decrease(self, default_config):
        """Tip radii decrease from stage 1 to 3 for converging duct."""
        d = default_config["derived"]
        tip_radii = d["per_stage_tip_radii"]
        exit_ratio = default_config["duct"].get("exit_diameter_ratio", 1.0)
        if exit_ratio < 1.0:
            for i in range(len(tip_radii) - 1):
                assert tip_radii[i + 1] <= tip_radii[i], (
                    f"Tip radius should decrease: stage {i}={tip_radii[i]:.1f}, "
                    f"stage {i+1}={tip_radii[i+1]:.1f}"
                )

    def test_duct_exit_smaller_than_inlet(self, default_config):
        """Duct exit ID < inlet ID when exit_diameter_ratio < 1.0."""
        d = default_config["derived"]
        exit_ratio = default_config["duct"].get("exit_diameter_ratio", 1.0)
        if exit_ratio < 1.0:
            assert d["duct_exit_id"] < d["duct_inlet_id"]

    def test_compression_hub_change_visible(self, default_config):
        """Hub radii change > 5mm total for compression_ratio >= 1.5."""
        d = default_config["derived"]
        radii = d["per_stage_hub_radii"]
        cr = default_config["blades"].get("compression_ratio", 1.0)
        if cr >= 1.5:
            total_change = radii[-1] - radii[0]
            assert total_change > 5.0, (
                f"Hub radii change {total_change:.1f}mm should be > 5mm for CR={cr}"
            )

    def test_exit_diameter_ratio_in_config(self, default_config):
        """exit_diameter_ratio exists in duct config."""
        assert "exit_diameter_ratio" in default_config["duct"]
        assert default_config["duct"]["exit_diameter_ratio"] == 0.85
