"""Sealed hub housing generation using CadQuery.

Creates two hub halves that enclose the motor, gearbox, and bearings.
Features:
  - Stepped outer profile reflecting per-stage hub radii (compression)
  - Bearing seats for concentric shaft/tube architecture
  - Motor pocket and mounting holes
  - Magnet pockets for magnetic coupling zones
  - Carrier plate mounting bolt holes
  - Assembly bolt holes for split-plane joining
"""

import math
from typing import Tuple

import cadquery as cq

from src.utils import bolt_circle_points
from src.magnetic_coupling import MagneticCoupling


class HubGenerator:
    """Generates hub housing geometry (two halves)."""

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]
        self.hub_od = self.derived["hub_od"]
        self.hub_length = self.derived["hub_length"]
        self.wall_t = config["hub"]["wall_thickness"]
        self.motor = config["motor"]
        self.positions = self.derived["part_positions"]
        self.hub_start_z = self.positions["hub_half_a"]
        self.per_stage_hub_radii = self.derived["per_stage_hub_radii"]
        self.blade_axial_width = self.derived["blade_axial_width"]
        self.coupling = MagneticCoupling(config)

    def generate_half_a(self) -> cq.Workplane:
        """Generate the motor-side hub half (Half A).

        Contains: motor pocket, shaft bearing, middle tube bearing,
        first part of stepped profile.
        """
        half_len = self.hub_length / 2

        # Build stepped outer shell
        hub = self._create_stepped_shell(0, half_len)

        # Hollow interior
        inner_r = self.hub_od / 2 - self.wall_t
        interior = (
            cq.Workplane("XY")
            .circle(inner_r)
            .extrude(half_len - self.wall_t)  # leave bottom wall
        )
        hub = hub.cut(interior)

        # Motor pocket
        motor_r = self.motor["body_diameter"] / 2 + 0.2  # clearance
        motor_depth = self.motor["body_length"]
        motor_pocket = (
            cq.Workplane("XY")
            .workplane(offset=self.wall_t)
            .circle(motor_r)
            .extrude(min(motor_depth, half_len - 2 * self.wall_t))
        )
        hub = hub.cut(motor_pocket)

        # Inner shaft bearing seat (685ZZ at bottom wall)
        shaft_bearing = self.config["bearings"]["inner_shaft"]
        shaft_bore_r = shaft_bearing["od"] / 2 + self.config["hub"]["bearing_press_fit_clearance"]
        bore = (
            cq.Workplane("XY")
            .circle(shaft_bore_r)
            .extrude(half_len)
        )
        hub = hub.cut(bore)

        # Middle tube bearing seat (6702ZZ)
        middle_bearing = self.config["bearings"]["middle_tube"]
        middle_seat_r = middle_bearing["od"] / 2 + self.config["hub"]["bearing_press_fit_clearance"]
        middle_seat = (
            cq.Workplane("XY")
            .workplane(offset=half_len - middle_bearing["width"])
            .circle(middle_seat_r)
            .extrude(middle_bearing["width"])
        )
        hub = hub.cut(middle_seat)

        # Motor mounting holes
        hub = self._add_motor_mount_holes(hub)

        # Assembly bolt holes
        hub = self._add_bolt_holes(hub, half_len)

        # Magnet pockets on outer wall (for coupling zones in this half)
        hub = self._add_magnet_pockets(hub, 0, half_len)

        return hub

    def generate_half_b(self) -> cq.Workplane:
        """Generate the output-side hub half (Half B).

        Contains: gear stage cavities, coupling magnet zones, output bearings,
        second part of stepped profile.
        """
        half_len = self.hub_length / 2

        # Build stepped outer shell
        hub = self._create_stepped_shell(half_len, self.hub_length)

        # Hollow interior
        inner_r = self.hub_od / 2 - self.wall_t
        interior = (
            cq.Workplane("XY")
            .circle(inner_r)
            .extrude(half_len - self.wall_t)
        )
        hub = hub.cut(interior)

        # Outer tube bearing seat (6805ZZ at end wall)
        outer_bearing = self.config["bearings"]["outer_tube"]
        outer_seat_r = outer_bearing["od"] / 2 + self.config["hub"]["bearing_press_fit_clearance"]
        outer_seat = (
            cq.Workplane("XY")
            .workplane(offset=half_len - outer_bearing["width"])
            .circle(outer_seat_r)
            .extrude(outer_bearing["width"])
        )
        hub = hub.cut(outer_seat)

        # Middle tube bearing seat (6702ZZ at split plane)
        middle_bearing = self.config["bearings"]["middle_tube"]
        middle_seat_r = middle_bearing["od"] / 2 + self.config["hub"]["bearing_press_fit_clearance"]
        middle_bore = (
            cq.Workplane("XY")
            .circle(middle_seat_r)
            .extrude(half_len)
        )
        hub = hub.cut(middle_bore)

        # Assembly bolt holes (matching half A)
        hub = self._add_bolt_holes(hub, half_len)

        # Magnet pockets on outer wall (for coupling zones in this half)
        hub = self._add_magnet_pockets(hub, half_len, self.hub_length)

        return hub

    def _create_stepped_shell(self, z_start_abs: float, z_end_abs: float) -> cq.Workplane:
        """Create outer shell with stepped profile for compression.

        The shell diameter increases at each blade stage zone based on
        per_stage_hub_radii. Between blade zones, the hub uses the
        maximum hub_od.

        Args:
            z_start_abs: Absolute Z start of this half (in hub coordinates)
            z_end_abs: Absolute Z end of this half
        """
        half_len = z_end_abs - z_start_abs
        base_r = self.hub_od / 2

        # Start with constant-OD cylinder
        shell = (
            cq.Workplane("XY")
            .circle(base_r)
            .extrude(half_len)
        )

        # Add stepped protrusions at blade stage zones
        num_stages = len(self.per_stage_hub_radii)
        for i in range(num_stages):
            stage_r = self.per_stage_hub_radii[i]
            if stage_r <= base_r:
                continue  # no protrusion needed

            # Get blade stage Z position in hub-local coordinates
            blade_z_global = self.positions.get(f"blade_ring_stage_{i+1}", None)
            if blade_z_global is None:
                continue

            blade_z_local = blade_z_global - self.hub_start_z - z_start_abs
            blade_z_end = blade_z_local + self.blade_axial_width

            # Clip to this half's range
            z0 = max(blade_z_local, 0)
            z1 = min(blade_z_end, half_len)
            if z1 <= z0:
                continue

            # Add a ring at the larger radius
            ring = (
                cq.Workplane("XY")
                .workplane(offset=z0)
                .circle(stage_r)
                .circle(base_r)
                .extrude(z1 - z0)
            )
            shell = shell.union(ring)

        return shell

    def _add_magnet_pockets(self, hub: cq.Workplane, z_start_abs: float,
                            z_end_abs: float) -> cq.Workplane:
        """Add magnet pockets to hub outer wall for magnetic coupling zones.

        Pockets are cut from the outside inward at each blade stage
        coupling zone that falls within this hub half.
        """
        num_stages = len(self.config["magnetic_coupling"]["stages"])

        for i in range(num_stages):
            blade_z_global = self.positions.get(f"blade_ring_stage_{i+1}", None)
            if blade_z_global is None:
                continue

            blade_z_local = blade_z_global - self.hub_start_z - z_start_abs
            blade_z_mid = blade_z_local + self.blade_axial_width / 2
            half_len = z_end_abs - z_start_abs

            # Only add pockets if this stage is in this half
            if blade_z_mid < 0 or blade_z_mid > half_len:
                continue

            pocket_spec = self.coupling.magnet_pocket_specs(i)
            n_pockets = pocket_spec["num_pockets"]
            pocket_d = pocket_spec["pocket_diameter"]
            pocket_depth = pocket_spec["pocket_depth"]
            coupling_r = pocket_spec["coupling_radius"]
            angular_spacing = pocket_spec["angular_spacing"]

            # Get hub radius at this stage for pocket placement
            stage_r = self.per_stage_hub_radii[i] if i < len(self.per_stage_hub_radii) else self.hub_od / 2

            for j in range(n_pockets):
                angle = math.radians(j * angular_spacing)
                cx = coupling_r * math.cos(angle)
                cy = coupling_r * math.sin(angle)

                # Pocket cut from outside, at blade stage z position
                pocket = (
                    cq.Workplane("XY")
                    .workplane(offset=blade_z_local)
                    .center(cx, cy)
                    .circle(pocket_d / 2)
                    .extrude(pocket_depth)
                )
                hub = hub.cut(pocket)

        return hub

    def _add_motor_mount_holes(self, hub: cq.Workplane) -> cq.Workplane:
        """Add motor mounting bolt holes."""
        pcd = self.motor["mounting_holes_pcd"]
        n_holes = self.motor["mounting_holes_count"]
        hole_d = self.motor["mounting_holes_diameter"]

        positions = bolt_circle_points(pcd, n_holes)
        for x, y in positions:
            hole = (
                cq.Workplane("XY")
                .center(x, y)
                .circle(hole_d / 2)
                .extrude(self.wall_t + 5)  # through wall
            )
            hub = hub.cut(hole)

        return hub

    def _add_bolt_holes(self, hub: cq.Workplane, length: float) -> cq.Workplane:
        """Add assembly bolt holes on the split plane face."""
        bolt_pcd = self.hub_od - 2 * self.wall_t - 4  # mm
        n_bolts = 6
        bolt_d = 3.0  # M3

        positions = bolt_circle_points(bolt_pcd, n_bolts)
        for x, y in positions:
            hole = (
                cq.Workplane("XY")
                .center(x, y)
                .circle(bolt_d / 2)
                .extrude(length)
            )
            hub = hub.cut(hole)

        return hub

    def generate_both_halves(self) -> Tuple[cq.Workplane, cq.Workplane]:
        """Generate both hub halves."""
        return self.generate_half_a(), self.generate_half_b()
