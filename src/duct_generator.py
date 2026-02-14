"""Duct shell generation with bellmouth inlet using CadQuery.

Creates the outer duct that encloses the fan stages. Features a curved
bellmouth inlet for reduced inlet losses. Supports converging duct
geometry where exit ID < inlet ID for visible compression effect.
Can be split into sections if the total length exceeds the build volume.

Uses extrude-based geometry (not revolve) to ensure watertight STL output.
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
        self.exit_diameter_ratio = self.duct_cfg.get("exit_diameter_ratio", 1.0)

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

        For converging ducts (exit_diameter_ratio < 1.0), creates a tapered
        inner wall using a revolved trapezoidal profile. The outer wall
        stays constant for structural strength.

        Args:
            z_offset: Axial start position
            length: Section length
            include_bellmouth: Whether to add bellmouth lip
        """
        outer_r = self.duct_od / 2
        inner_r_inlet = self.duct_id / 2
        inner_r_exit = self.duct_id * self.exit_diameter_ratio / 2
        br = self.bellmouth_r if include_bellmouth else 0

        is_converging = abs(self.exit_diameter_ratio - 1.0) > 0.01

        if is_converging:
            # Converging duct: revolved trapezoidal XZ profile
            # Outer wall constant, inner wall tapers from inlet ID to exit ID
            duct = (
                cq.Workplane("XZ")
                .moveTo(inner_r_inlet, 0)
                .lineTo(outer_r, 0)
                .lineTo(outer_r, length)
                .lineTo(inner_r_exit, length)
                .close()
                .revolve(360, (0, 0, 0), (0, 1, 0))
            )
        else:
            # Straight duct: extruded annulus (always watertight)
            duct = (
                cq.Workplane("XY")
                .circle(outer_r)
                .circle(inner_r_inlet)
                .extrude(length)
            )

        if br > 0:
            # Bellmouth flare extending below z=0.
            # Strategy: extrude a wider annular ring, then cut away the
            # outer taper with a revolved cone shape.
            bellmouth_ring = (
                cq.Workplane("XY")
                .workplane(offset=-br)
                .circle(outer_r + br)
                .circle(inner_r_inlet)
                .extrude(br)
            )

            # Cone taper to cut: profile in XZ revolved around Z.
            # This removes the outer corner, creating the flared lip shape.
            taper_margin = 10.0  # extra radial extent to ensure clean cut
            taper_cut = (
                cq.Workplane("XZ")
                .moveTo(outer_r, 0)
                .lineTo(outer_r + br, -br)
                .lineTo(outer_r + br + taper_margin, -br)
                .lineTo(outer_r + br + taper_margin, 0)
                .close()
                .revolve(360, (0, 0, 0), (0, 1, 0))
            )

            bellmouth_ring = bellmouth_ring.cut(taper_cut)
            duct = duct.union(bellmouth_ring)

        # Add stator mounting slots (use inlet radius for entry, exit radius for exit)
        duct = self._add_stator_slots(duct, inner_r_inlet, inner_r_exit, length)

        return duct

    def _add_stator_slots(self, duct: cq.Workplane, inner_r_inlet: float,
                          inner_r_exit: float, length: float) -> cq.Workplane:
        """Add mounting slots for stator struts on the inner surface.

        Uses cylindrical slot geometry instead of rectangular boxes to
        avoid non-manifold edges on the curved duct wall.
        Slot positions are derived from the axial layout system.
        """
        n_struts = self.config["stators"]["num_struts"]
        strut_t = self.config["stators"]["strut_thickness"]
        slot_r = (strut_t + 0.4) / 2  # radius = half of (strut + clearance)
        slot_depth = 2.0  # mm into duct wall
        slot_height = self.config["stators"]["strut_chord"] + 1.0

        # Get stator positions from layout and convert to duct-local Z
        positions = self.derived.get("part_positions", {})
        duct_z_offset = positions.get("duct_section_1", 0)
        entry_z = positions.get("stator_entry", 10) - duct_z_offset
        exit_z = positions.get("stator_exit", length - 10 - slot_height) - duct_z_offset

        stator_configs = [
            (entry_z, inner_r_inlet),
            (exit_z, inner_r_exit),
        ]

        for z_pos, local_inner_r in stator_configs:
            if z_pos < 0 or z_pos + slot_height > length:
                continue
            for i in range(n_struts):
                angle = 360.0 * i / n_struts
                angle_rad = math.radians(angle)

                cx = (local_inner_r + slot_depth / 2) * math.cos(angle_rad)
                cy = (local_inner_r + slot_depth / 2) * math.sin(angle_rad)

                slot = (
                    cq.Workplane("XY")
                    .workplane(offset=z_pos)
                    .pushPoints([(cx, cy)])
                    .circle(slot_r)
                    .extrude(slot_height)
                )
                duct = duct.cut(slot)

        return duct
