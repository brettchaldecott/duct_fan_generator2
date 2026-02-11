"""Duct shell generation with bellmouth inlet using CadQuery.

Creates the outer duct that encloses the fan stages. Features a curved
bellmouth inlet for reduced inlet losses. Can be split into sections
if the total length exceeds the build volume.
"""

import math

import cadquery as cq


class DuctGenerator:
    """Generates duct shell geometry."""

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]
        self.duct_cfg = config["duct"]
        self.print_cfg = config["print"]

        self.duct_id = self.duct_cfg["inner_diameter"]
        self.wall_t = self.duct_cfg["wall_thickness"]
        self.duct_od = self.duct_id + 2 * self.wall_t
        self.duct_length = self.derived["duct_length"]
        self.bellmouth_r = self.duct_cfg["bellmouth_radius"]

    def generate(self) -> list:
        """Generate duct sections (split if needed for build volume).

        Returns:
            List of cq.Workplane objects (one per section)
        """
        max_z = self.print_cfg["max_build_z"]

        if self.duct_length <= max_z:
            return [self._create_duct_section(0, self.duct_length, include_bellmouth=True)]
        else:
            # Split into sections
            sections = []
            n_sections = math.ceil(self.duct_length / (max_z - 10))  # 10mm overlap zone
            section_len = self.duct_length / n_sections

            for i in range(n_sections):
                z_start = i * section_len
                z_end = (i + 1) * section_len
                bellmouth = (i == 0)  # only first section has bellmouth
                sections.append(
                    self._create_duct_section(z_start, z_end - z_start,
                                              include_bellmouth=bellmouth)
                )

            return sections

    def _create_duct_section(self, z_offset: float, length: float,
                             include_bellmouth: bool = False) -> cq.Workplane:
        """Create a single duct section.

        Args:
            z_offset: Axial start position
            length: Section length
            include_bellmouth: Whether to add bellmouth lip
        """
        outer_r = self.duct_od / 2
        inner_r = self.duct_id / 2

        # Main cylindrical shell
        duct = (
            cq.Workplane("XY")
            .circle(outer_r)
            .circle(inner_r)
            .extrude(length)
        )

        # Add bellmouth inlet lip
        if include_bellmouth and self.bellmouth_r > 0:
            duct = self._add_bellmouth(duct, outer_r, inner_r)

        # Add stator mounting slots
        duct = self._add_stator_slots(duct, inner_r, length)

        return duct

    def _add_bellmouth(self, duct: cq.Workplane, outer_r: float,
                       inner_r: float) -> cq.Workplane:
        """Add curved bellmouth inlet lip to the duct entrance.

        Creates a flared lip using a revolved profile for smooth air entry.
        """
        br = self.bellmouth_r

        # Create a simple flared lip by revolving a triangular/curved profile
        # Profile in XZ plane: starts at (outer_r, 0), curves to (outer_r + br, -br)
        lip = (
            cq.Workplane("XZ")
            .moveTo(inner_r, 0)
            .lineTo(inner_r - br, -br)
            .lineTo(outer_r, -br)
            .lineTo(outer_r, 0)
            .close()
            .revolve(360, (0, 0, 0), (0, 0, 1))
        )

        duct = duct.union(lip)
        return duct

    def _add_stator_slots(self, duct: cq.Workplane, inner_r: float,
                          length: float) -> cq.Workplane:
        """Add mounting slots for stator struts on the inner surface."""
        n_struts = self.config["stators"]["num_struts"]
        slot_width = self.config["stators"]["strut_thickness"] + 0.4  # clearance
        slot_depth = 2.0  # mm into duct wall
        slot_height = self.config["stators"]["strut_chord"] + 1.0

        # Stator positions: entry and exit
        for z_pos in [10, length - 10 - slot_height]:
            if z_pos < 0:
                continue
            for i in range(n_struts):
                angle = 360.0 * i / n_struts
                angle_rad = math.radians(angle)

                cx = (inner_r + slot_depth / 2) * math.cos(angle_rad)
                cy = (inner_r + slot_depth / 2) * math.sin(angle_rad)

                slot = (
                    cq.Workplane("XY")
                    .workplane(offset=z_pos)
                    .center(cx, cy)
                    .rect(slot_depth, slot_width)
                    .extrude(slot_height)
                )
                duct = duct.cut(slot)

        return duct
