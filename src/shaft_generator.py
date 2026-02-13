"""Concentric shaft and tube generation using CadQuery.

Creates the inner shaft, middle tube, and outer tube that form the
concentric drive train. Each element connects a gear stage output
to its corresponding blade stage.

Also generates coupling discs that connect drive elements radially
out to the hub wall at each blade stage coupling zone.

Drive train:
  Inner shaft (5mm) — motor to sun gear stage 0 — drives blade stage 1
  Middle tube (15mm OD) — ring gear stage 0 output — drives blade stage 2
  Outer tube (25mm OD) — ring gear stage 1 output — drives blade stage 3
"""

import math
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
        start_z = self.derived.get("inner_shaft_start_z", 0)
        end_z = self.derived.get("inner_shaft_end_z", self.hub_length)
        length = end_z - start_z

        shaft = (
            cq.Workplane("XY")
            .circle(shaft_d / 2)
            .extrude(length)
        )

        return shaft

    def generate_middle_tube(self) -> cq.Workplane:
        """Generate the middle tube.

        Connects ring gear stage 0 output to blade stage 2.
        Only spans from after gear stage 0 to gear stage 1 region.
        """
        od = self.derived["middle_tube_od"]
        tube_id = self.derived["middle_tube_id"]
        start_z = self.derived.get("middle_tube_start_z", 0)
        end_z = self.derived.get("middle_tube_end_z", self.hub_length)
        length = end_z - start_z

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
        Only spans from after gear stage 1 to end of hub.
        """
        od = self.derived["outer_tube_od"]
        tube_id = self.derived["outer_tube_id"]
        start_z = self.derived.get("outer_tube_start_z", 0)
        end_z = self.derived.get("outer_tube_end_z", self.hub_length)
        length = end_z - start_z

        tube = (
            cq.Workplane("XY")
            .circle(od / 2)
            .circle(tube_id / 2)
            .extrude(length)
        )

        return tube

    def generate_coupling_disc(self, stage_index: int) -> cq.Workplane:
        """Generate a coupling disc for a blade stage.

        Connects the drive element (shaft or tube) radially outward
        to the hub wall at the blade stage coupling zone.

        Stage 1: inner shaft (5mm) -> hub_r
        Stage 2: middle tube (15mm OD) -> hub_r
        Stage 3: outer tube (25mm OD) -> hub_r

        Includes clearance holes for inner concentric elements
        and lightening holes at mid-radius.
        """
        per_stage_hub_radii = self.derived["per_stage_hub_radii"]
        outer_r = per_stage_hub_radii[stage_index]
        disc_thickness = 5.0  # mm

        shaft_d = self.derived["inner_shaft_diameter"]
        middle_od = self.derived["middle_tube_od"]
        outer_od = self.derived["outer_tube_od"]

        if stage_index == 0:
            # Stage 1: driven by inner shaft
            inner_r = shaft_d / 2
            clearance_bore_r = 0  # nothing inside the shaft
        elif stage_index == 1:
            # Stage 2: driven by middle tube
            inner_r = middle_od / 2
            clearance_bore_r = shaft_d / 2 + 0.5  # clearance for inner shaft
        else:
            # Stage 3: driven by outer tube
            inner_r = outer_od / 2
            clearance_bore_r = middle_od / 2 + 0.5  # clearance for middle tube

        # Create disc: outer radius to inner radius (drive element)
        disc = (
            cq.Workplane("XY")
            .workplane(offset=-disc_thickness / 2)
            .circle(outer_r)
            .circle(inner_r)
            .extrude(disc_thickness)
        )

        # Add clearance bore for inner concentric elements
        if clearance_bore_r > 0 and clearance_bore_r < inner_r:
            # Already cleared by the inner circle
            pass
        elif clearance_bore_r > 0:
            bore = (
                cq.Workplane("XY")
                .workplane(offset=-disc_thickness / 2)
                .circle(clearance_bore_r)
                .extrude(disc_thickness)
            )
            disc = disc.cut(bore)

        # Add 4 lightening holes at mid-radius
        mid_r = (inner_r + outer_r) / 2
        hole_r = min((outer_r - inner_r) * 0.2, 5.0)  # 20% of radial span or 5mm max
        if hole_r >= 2.0:  # only add if meaningful size
            for i in range(4):
                angle = math.radians(45 + 90 * i)
                cx = mid_r * math.cos(angle)
                cy = mid_r * math.sin(angle)
                hole = (
                    cq.Workplane("XY")
                    .workplane(offset=-disc_thickness / 2)
                    .center(cx, cy)
                    .circle(hole_r)
                    .extrude(disc_thickness)
                )
                disc = disc.cut(hole)

        return disc

    def generate_all(self) -> Dict[str, cq.Workplane]:
        """Generate all shafts, tubes, and coupling discs.

        Returns:
            Dict of part_name -> CadQuery Workplane
        """
        parts = {
            "inner_shaft": self.generate_inner_shaft(),
            "middle_tube": self.generate_middle_tube(),
            "outer_tube": self.generate_outer_tube(),
        }

        # Generate coupling discs for each blade stage
        num_stages = len(self.config["blades"]["stages"])
        for i in range(num_stages):
            parts[f"coupling_disc_stage_{i+1}"] = self.generate_coupling_disc(i)

        return parts
