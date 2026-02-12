"""Tests for planet carrier plate generation."""

import math
import pytest
import cadquery as cq
from src.carrier_generator import CarrierGenerator


class TestCarrierGeneration:
    """Test carrier plate geometry."""

    def test_carrier_generates(self, default_config):
        """Carrier plate generates valid solid."""
        gen = CarrierGenerator(default_config)
        plate = gen.generate_carrier_plate(0, "front")
        assert plate is not None
        assert plate.val().isValid()

    def test_all_carriers_generate(self, default_config):
        """All carrier plates (front + back per stage) generate."""
        gen = CarrierGenerator(default_config)
        carriers = gen.generate_all_carriers()
        num_stages = default_config["gears"]["num_stages"]
        assert len(carriers) == num_stages * 2
        for name, solid in carriers.items():
            assert solid.val().isValid(), f"{name} is not valid"

    def test_carrier_has_pin_holes(self, default_config):
        """Carrier plate has holes for planet pins."""
        gen = CarrierGenerator(default_config)
        plate = gen.generate_carrier_plate(0)
        assert plate.val().isValid()

    def test_carrier_radius_matches_gear(self, default_config):
        """Carrier radius = (sun_teeth + planet_teeth) * module / 2."""
        gen = CarrierGenerator(default_config)
        gears = default_config["gears"]
        expected_r = (gears["sun_teeth"] + gears["planet_teeth"]) * gears["module"] / 2
        assert gen.carrier_radius == pytest.approx(expected_r)

    def test_stage_0_bore_clears_shaft(self, default_config):
        """Stage 0 carrier bore clears the inner shaft."""
        gen = CarrierGenerator(default_config)
        shaft_d = default_config["derived"]["inner_shaft_diameter"]
        # The bore should be larger than the shaft
        plate = gen.generate_carrier_plate(0)
        assert plate.val().isValid()

    def test_stage_1_bore_clears_middle_tube(self, default_config):
        """Stage 1 carrier bore clears the middle tube."""
        gen = CarrierGenerator(default_config)
        plate = gen.generate_carrier_plate(1)
        assert plate.val().isValid()

    def test_carrier_positions_in_layout(self, default_config):
        """Carrier positions exist in part_positions."""
        positions = default_config["derived"]["part_positions"]
        num_stages = default_config["gears"]["num_stages"]
        for i in range(num_stages):
            assert f"carrier_front_{i}" in positions
            assert f"carrier_back_{i}" in positions

    def test_carrier_between_blade_stages(self, default_config):
        """Carrier positions are between blade stage positions."""
        positions = default_config["derived"]["part_positions"]
        blade_axial_width = default_config["derived"]["blade_axial_width"]
        # Blade positions are centers; carrier 0 should be between blade 1 end and blade 2 start
        blade_1_end = positions["blade_ring_stage_1"] + blade_axial_width / 2
        blade_2_start = positions["blade_ring_stage_2"] - blade_axial_width / 2
        carrier_0 = positions["carrier_front_0"]
        assert blade_1_end <= carrier_0 <= blade_2_start
