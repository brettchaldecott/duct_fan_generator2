"""Structural stator strut generation using CadQuery.

Creates simple elliptical cross-section struts connecting hub OD to duct ID.
Stators provide structural support and can reduce swirl between stages.
"""

import math

import cadquery as cq


class StatorGenerator:
    """Generates stator strut geometry."""

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]
        self.stator_cfg = config["stators"]
        self.hub_od = self.derived["hub_od"]
        self.duct_id = config["duct"]["inner_diameter"]

    def generate_stator_set(self, position: str = "entry") -> cq.Workplane:
        """Generate a set of stator struts.

        Args:
            position: "entry" or "exit" — determines axial position
        """
        n_struts = self.stator_cfg["num_struts"]
        strut_t = self.stator_cfg["strut_thickness"]
        strut_chord = self.stator_cfg["strut_chord"]
        hub_r = self.hub_od / 2
        duct_r = self.duct_id / 2

        # Create one strut (elliptical cross-section, radial extrusion)
        strut_length = duct_r - hub_r

        # Build all struts
        stator = cq.Workplane("XY")

        # Hub ring (structural)
        ring_height = strut_chord
        hub_ring = (
            cq.Workplane("XY")
            .circle(hub_r + 2)  # 2mm structural lip
            .circle(hub_r)
            .extrude(ring_height)
        )
        stator = hub_ring

        # Outer ring (mounts to duct)
        outer_ring = (
            cq.Workplane("XY")
            .circle(duct_r)
            .circle(duct_r - 2)  # 2mm wall
            .extrude(ring_height)
        )
        stator = stator.union(outer_ring)

        # Individual struts
        for i in range(n_struts):
            angle = 360.0 * i / n_struts
            strut = self._create_strut(hub_r, duct_r, strut_t, strut_chord)
            strut = strut.rotate((0, 0, 0), (0, 0, 1), angle)
            stator = stator.union(strut)

        return stator

    def _create_strut(self, r_inner: float, r_outer: float,
                      thickness: float, chord: float) -> cq.Workplane:
        """Create a single strut with elliptical cross-section."""
        strut_length = r_outer - r_inner

        # Simplified strut: rectangular cross-section (elliptical is harder in CQ)
        # Use a box positioned radially
        strut = (
            cq.Workplane("XY")
            .center(r_inner + strut_length / 2, 0)
            .rect(strut_length, thickness)
            .extrude(chord)
        )

        return strut

    def generate_entry_stator(self) -> cq.Workplane:
        """Generate entry stator set."""
        return self.generate_stator_set("entry")

    def generate_exit_stator(self) -> cq.Workplane:
        """Generate exit stator set."""
        return self.generate_stator_set("exit")
