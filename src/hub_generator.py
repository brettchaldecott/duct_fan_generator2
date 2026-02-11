"""Sealed hub housing generation using CadQuery.

Creates two hub halves that enclose the motor, gearbox, and bearings.
Includes motor pocket, bearing seats, gearbox cavities, magnet ring
zones, and bolt holes for assembly.
"""

import math
from typing import Tuple

import cadquery as cq

from src.utils import bolt_circle_points


class HubGenerator:
    """Generates hub housing geometry (two halves)."""

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]
        self.hub_od = self.derived["hub_od"]
        self.hub_length = self.derived["hub_length"]
        self.wall_t = config["hub"]["wall_thickness"]
        self.motor = config["motor"]

    def generate_half_a(self) -> cq.Workplane:
        """Generate the motor-side hub half (Half A).

        Contains: motor pocket, shaft bearing, first gear stage cavity.
        """
        half_len = self.hub_length / 2

        # Outer shell
        hub = (
            cq.Workplane("XY")
            .circle(self.hub_od / 2)
            .extrude(half_len)
        )

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

        # Shaft bore
        shaft_bearing = self.config["bearings"]["inner_shaft"]
        shaft_bore_r = shaft_bearing["od"] / 2 + self.config["hub"]["bearing_press_fit_clearance"]
        bore = (
            cq.Workplane("XY")
            .circle(shaft_bore_r)
            .extrude(half_len)
        )
        hub = hub.cut(bore)

        # Motor mounting holes
        hub = self._add_motor_mount_holes(hub)

        # Assembly bolt holes
        hub = self._add_bolt_holes(hub, half_len)

        return hub

    def generate_half_b(self) -> cq.Workplane:
        """Generate the output-side hub half (Half B).

        Contains: gear stage cavities, coupling magnet zones, output bearings.
        """
        half_len = self.hub_length / 2

        # Outer shell
        hub = (
            cq.Workplane("XY")
            .circle(self.hub_od / 2)
            .extrude(half_len)
        )

        # Hollow interior
        inner_r = self.hub_od / 2 - self.wall_t
        interior = (
            cq.Workplane("XY")
            .circle(inner_r)
            .extrude(half_len - self.wall_t)
        )
        hub = hub.cut(interior)

        # Output bearing seat
        outer_bearing = self.config["bearings"]["outer_tube"]
        bearing_r = outer_bearing["od"] / 2 + self.config["hub"]["bearing_press_fit_clearance"]
        bearing_seat = (
            cq.Workplane("XY")
            .workplane(offset=half_len - outer_bearing["width"])
            .circle(bearing_r)
            .extrude(outer_bearing["width"])
        )
        hub = hub.cut(bearing_seat)

        # Central bore for tube/shaft
        middle_bearing = self.config["bearings"]["middle_tube"]
        bore_r = middle_bearing["od"] / 2 + self.config["hub"]["bearing_press_fit_clearance"]
        bore = (
            cq.Workplane("XY")
            .circle(bore_r)
            .extrude(half_len)
        )
        hub = hub.cut(bore)

        # Assembly bolt holes (matching half A)
        hub = self._add_bolt_holes(hub, half_len)

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
        # Bolt holes on a circle just inside the OD
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
