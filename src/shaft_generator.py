"""Concentric shaft and tube generation using CadQuery.

Creates the inner shaft, middle tube, and outer tube that form the
concentric drive train. Each element connects a gear stage output
to its corresponding blade stage.

Drive train:
  Inner shaft (5mm) — motor to sun gear stage 0 — drives blade stage 1
  Middle tube (15mm OD) — ring gear stage 0 output — drives blade stage 2
  Outer tube (25mm OD) — ring gear stage 1 output — drives blade stage 3
"""

from typing import Dict

import cadquery as cq


class ShaftGenerator:
    """Generates concentric shaft and tube geometry."""

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]
        self.hub_length = self.derived["hub_length"]

    def generate_inner_shaft(self) -> cq.Workplane:
        """Generate the inner shaft (motor shaft extension).

        5mm diameter, runs the full hub length.
        Connects motor to sun gear stage 0.
        """
        shaft_d = self.derived["inner_shaft_diameter"]
        length = self.hub_length

        shaft = (
            cq.Workplane("XY")
            .circle(shaft_d / 2)
            .extrude(length)
        )

        return shaft

    def generate_middle_tube(self) -> cq.Workplane:
        """Generate the middle tube.

        Connects ring gear stage 0 output to blade stage 2.
        OD = middle_tube_od, ID = middle_tube_id.
        """
        od = self.derived["middle_tube_od"]
        tube_id = self.derived["middle_tube_id"]
        length = self.hub_length

        tube = (
            cq.Workplane("XY")
            .circle(od / 2)
            .circle(tube_id / 2)
            .extrude(length)
        )

        return tube

    def generate_outer_tube(self) -> cq.Workplane:
        """Generate the outer tube.

        Connects ring gear stage 1 output to blade stage 3.
        OD = outer_tube_od, ID = outer_tube_id.
        """
        od = self.derived["outer_tube_od"]
        tube_id = self.derived["outer_tube_id"]
        length = self.hub_length

        tube = (
            cq.Workplane("XY")
            .circle(od / 2)
            .circle(tube_id / 2)
            .extrude(length)
        )

        return tube

    def generate_all(self) -> Dict[str, cq.Workplane]:
        """Generate all shafts and tubes.

        Returns:
            Dict of part_name -> CadQuery Workplane
        """
        return {
            "inner_shaft": self.generate_inner_shaft(),
            "middle_tube": self.generate_middle_tube(),
            "outer_tube": self.generate_outer_tube(),
        }
