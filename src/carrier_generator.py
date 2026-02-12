"""Planet carrier plate generation using CadQuery.

Creates fixed carrier plates that hold planet gear pins in position.
Each gear stage has front and back carrier plates. The carrier is
fixed to the hub housing (bolted or press-fit).
"""

import math
from typing import Dict

import cadquery as cq

from src.utils import bolt_circle_points


class CarrierGenerator:
    """Generates planet carrier plate geometry."""

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]
        self.gears = config["gears"]
        self.module = self.gears["module"]
        self.num_planets = self.gears["num_planets"]

        # Carrier radius = center distance from sun to planet
        sun_teeth = self.gears["sun_teeth"]
        planet_teeth = self.gears["planet_teeth"]
        self.carrier_radius = (sun_teeth + planet_teeth) * self.module / 2

        # Planet pin diameter (sized for bearing or press-fit)
        self.pin_diameter = self.config["motor"]["shaft_diameter"]  # 5mm

        # Carrier plate dimensions
        self.plate_thickness = 3.0  # mm
        self.hub_inner_r = self.derived["hub_od"] / 2 - self.config["hub"]["wall_thickness"]

    def generate_carrier_plate(self, stage_index: int, position: str = "front") -> cq.Workplane:
        """Generate a carrier plate for a gear stage.

        Args:
            stage_index: 0-based gear stage index
            position: "front" or "back"

        Returns:
            CadQuery Workplane with the carrier plate solid
        """
        # Determine central bore size based on which shafts/tubes pass through
        if stage_index == 0:
            # Stage 0: inner shaft passes through
            bore_d = self.derived["inner_shaft_diameter"] + 1.0  # clearance
        else:
            # Stage 1+: middle tube passes through
            bore_d = self.derived["middle_tube_od"] + 1.0  # clearance

        outer_r = self.hub_inner_r - 0.5  # slight clearance to hub wall
        bore_r = bore_d / 2

        # Annular plate
        plate = (
            cq.Workplane("XY")
            .circle(outer_r)
            .circle(bore_r)
            .extrude(self.plate_thickness)
        )

        # Planet pin holes at carrier radius
        pin_r = self.pin_diameter / 2 + 0.1  # slight clearance for pin
        angles = [2 * math.pi * i / self.num_planets for i in range(self.num_planets)]
        for angle in angles:
            cx = self.carrier_radius * math.cos(angle)
            cy = self.carrier_radius * math.sin(angle)
            pin_hole = (
                cq.Workplane("XY")
                .center(cx, cy)
                .circle(pin_r)
                .extrude(self.plate_thickness)
            )
            plate = plate.cut(pin_hole)

        # Mounting holes to hub housing (bolt circle near outer edge)
        bolt_pcd = outer_r * 2 - 8  # 4mm from edge
        n_bolts = 4
        bolt_r = 1.5  # M3 through-hole
        bolt_positions = bolt_circle_points(bolt_pcd, n_bolts, start_angle=15)
        for bx, by in bolt_positions:
            bolt_hole = (
                cq.Workplane("XY")
                .center(bx, by)
                .circle(bolt_r)
                .extrude(self.plate_thickness)
            )
            plate = plate.cut(bolt_hole)

        return plate

    def generate_all_carriers(self) -> Dict[str, cq.Workplane]:
        """Generate all carrier plates for all gear stages.

        Returns:
            Dict of part_name -> CadQuery Workplane
        """
        num_stages = self.gears["num_stages"]
        solids = {}

        for i in range(num_stages):
            front = self.generate_carrier_plate(i, "front")
            back = self.generate_carrier_plate(i, "back")
            solids[f"carrier_front_{i}"] = front
            solids[f"carrier_back_{i}"] = back

        return solids
