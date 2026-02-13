"""Tests for concentric shaft and tube generation."""

import pytest
import cadquery as cq
from src.shaft_generator import ShaftGenerator


class TestShaftGeneration:
    """Test shaft and tube geometry."""

    def test_inner_shaft_generates(self, default_config):
        """Inner shaft generates valid solid."""
        gen = ShaftGenerator(default_config)
        shaft = gen.generate_inner_shaft()
        assert shaft is not None
        assert shaft.val().isValid()

    def test_middle_tube_generates(self, default_config):
        """Middle tube generates valid solid."""
        gen = ShaftGenerator(default_config)
        tube = gen.generate_middle_tube()
        assert tube is not None
        assert tube.val().isValid()

    def test_outer_tube_generates(self, default_config):
        """Outer tube generates valid solid."""
        gen = ShaftGenerator(default_config)
        tube = gen.generate_outer_tube()
        assert tube is not None
        assert tube.val().isValid()

    def test_all_shafts_generate(self, default_config):
        """All shafts, tubes, and coupling discs generate."""
        gen = ShaftGenerator(default_config)
        parts = gen.generate_all()
        num_stages = len(default_config["blades"]["stages"])
        # 3 shafts/tubes + N coupling discs
        assert len(parts) == 3 + num_stages
        assert "inner_shaft" in parts
        assert "middle_tube" in parts
        assert "outer_tube" in parts
        for i in range(num_stages):
            assert f"coupling_disc_stage_{i+1}" in parts
        for name, solid in parts.items():
            assert solid.val().isValid(), f"{name} is not valid"


class TestConcentricDimensions:
    """Test that shafts/tubes are properly sized for nesting."""

    def test_inner_shaft_fits_in_middle_tube(self, default_config):
        """Inner shaft diameter < middle tube ID."""
        d = default_config["derived"]
        assert d["inner_shaft_diameter"] < d["middle_tube_id"]

    def test_middle_tube_fits_in_outer_tube(self, default_config):
        """Middle tube OD < outer tube ID."""
        d = default_config["derived"]
        assert d["middle_tube_od"] < d["outer_tube_id"]

    def test_shaft_dimensions_positive(self, default_config):
        """All shaft/tube dimensions are positive."""
        d = default_config["derived"]
        assert d["inner_shaft_diameter"] > 0
        assert d["middle_tube_od"] > 0
        assert d["middle_tube_id"] > 0
        assert d["outer_tube_od"] > 0
        assert d["outer_tube_id"] > 0

    def test_tube_wall_thickness_positive(self, default_config):
        """Tube wall thickness (OD - ID) / 2 is positive."""
        d = default_config["derived"]
        middle_wall = (d["middle_tube_od"] - d["middle_tube_id"]) / 2
        outer_wall = (d["outer_tube_od"] - d["outer_tube_id"]) / 2
        assert middle_wall > 0, f"Middle tube wall={middle_wall}mm"
        assert outer_wall > 0, f"Outer tube wall={outer_wall}mm"
