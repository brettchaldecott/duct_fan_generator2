"""3D blade ring geometry generation using CadQuery.

Creates blade rings from BEMT-optimized airfoil sections, lofted along
the radial span. Includes ring bore, magnet pockets, and bearing seat.
"""

import math
import numpy as np
from typing import List, Tuple, Optional

import cadquery as cq

from src.airfoil import generate_naca4_profile, blend_profiles
from src.magnetic_coupling import MagneticCoupling


class BladeRingGenerator:
    """Generates 3D blade ring geometry for a single stage."""

    def __init__(self, config: dict, stage_index: int, bemt_stage=None):
        self.config = config
        self.stage_index = stage_index
        self.derived = config["derived"]
        self.blade_cfg = config["blades"]["stages"][stage_index]
        self.coupling = MagneticCoupling(config)
        self.bemt_stage = bemt_stage

        # Use per-stage hub radius (hub housing outer wall)
        per_stage = self.derived.get("per_stage_hub_radii", None)
        if per_stage and stage_index < len(per_stage):
            self.hub_r = per_stage[stage_index]
        else:
            self.hub_r = self.derived["blade_hub_radius"]  # mm

        # External blade ring geometry: ring wraps OUTSIDE hub with air gap
        blade_ring_radii = self.derived.get("blade_ring_radii", None)
        if blade_ring_radii and stage_index < len(blade_ring_radii):
            self.ring_inner_r = blade_ring_radii[stage_index]["ring_inner_r"]
            self.ring_outer_r = blade_ring_radii[stage_index]["ring_outer_r"]
        else:
            # Fallback: ring at hub surface
            air_gap = self.derived.get("blade_ring_air_gap", 1.0)
            wall_t = config["hub"]["wall_thickness"]
            self.ring_inner_r = self.hub_r + air_gap
            self.ring_outer_r = self.ring_inner_r + wall_t

        self.tip_r = self.derived["blade_tip_radius"]   # mm
        self.n_blades = self.blade_cfg["num_blades"]
        self.direction = self.derived["stage_directions"][stage_index]

    def generate(self) -> cq.Workplane:
        """Generate complete blade ring: hub ring + blades + magnet pockets."""
        # Create the hub ring (annular bore)
        ring = self._create_hub_ring()

        # Add blades
        for i in range(self.n_blades):
            angle = 360.0 * i / self.n_blades
            blade = self._create_blade(i)
            ring = ring.union(blade.rotate((0, 0, 0), (0, 0, 1), angle))

        # Add magnet pockets (subtract)
        ring = self._add_magnet_pockets(ring)

        return ring

    def _create_hub_ring(self) -> cq.Workplane:
        """Create the blade ring that wraps OUTSIDE the hub housing.

        The ring inner surface faces the hub outer wall across an air gap
        (magnetic coupling interface). Blades attach to the ring outer surface.
        """
        ring_height = self.derived.get("blade_axial_width", 25)

        # Ring centered at Z=0 for symmetric axial positioning
        ring = (
            cq.Workplane("XY")
            .workplane(offset=-ring_height / 2)
            .circle(self.ring_outer_r)
            .circle(self.ring_inner_r)
            .extrude(ring_height)
        )

        return ring

    def _create_blade(self, blade_index: int) -> cq.Workplane:
        """Create a single blade via airfoil section lofting."""
        n_sections = self.config["blades"]["num_radial_sections"]
        root_designation = self.blade_cfg["airfoil_root"]
        tip_designation = self.blade_cfg["airfoil_tip"]

        # Duct constraint: max radial reach for mid-chord centered blade
        duct_ir = self.config["duct"]["inner_diameter"] / 2  # mm
        tip_cl = self.config["blades"]["tip_clearance"]  # mm
        max_r = duct_ir - tip_cl

        # Radial stations — start slightly inside ring OD for clean boolean union
        BLADE_ROOT_OVERLAP = 1.0  # mm overlap into ring for watertight union
        blade_root_r = self.ring_outer_r - BLADE_ROOT_OVERLAP
        radii = np.linspace(blade_root_r, self.tip_r, n_sections)

        # Generate sections, skipping degenerate ones near tip
        MIN_CHORD = 3.0  # mm — minimum viable chord for loft
        sections = []
        for j, r in enumerate(radii):
            frac = max(0.0, (r - self.ring_outer_r) / (self.tip_r - self.ring_outer_r))

            # Get chord and twist from BEMT if available
            if self.bemt_stage and j < len(self.bemt_stage.sections):
                sec = self.bemt_stage.sections[min(j, len(self.bemt_stage.sections) - 1)]
                chord = sec.chord * 1000  # m to mm
                twist = math.degrees(sec.twist)
            else:
                # Fallback: linear distribution
                chord = 30 * (1 - 0.4 * frac)  # taper from 30mm to 18mm
                twist = 45 * (1 - frac) + 10 * frac  # 45° root to 10° tip

            # Duct-aware chord clamp: with mid-chord centering, max tangential
            # extent is chord/2, giving radial reach sqrt(r² + (chord/2)²).
            max_chord_at_r = 2 * math.sqrt(max(max_r**2 - r**2, 0))
            chord = min(chord, max_chord_at_r)
            chord = min(chord, 80)  # absolute maximum 80mm
            chord = max(chord, MIN_CHORD)

            # Skip this section if chord exceeds duct constraint at minimum
            if max_chord_at_r < MIN_CHORD:
                continue

            # Generate blended airfoil profile
            n_pts = 30
            root_profile = generate_naca4_profile(root_designation, n_pts, chord)
            tip_profile = generate_naca4_profile(tip_designation, n_pts, chord)
            profile = blend_profiles(root_profile, tip_profile, frac)

            sections.append((r, profile, twist))

        # Build CadQuery loft
        blade = self._loft_blade(sections)

        # Mirror for CCW stages
        if self.direction < 0:
            blade = blade.mirror("XZ")

        return blade

    @staticmethod
    def _dedup_points(points, tol=1e-4):
        """Remove consecutive duplicate 2D points within tolerance."""
        if not points:
            return points
        result = [points[0]]
        for pt in points[1:]:
            dx = pt[0] - result[-1][0]
            dy = pt[1] - result[-1][1]
            if (dx * dx + dy * dy) > tol * tol:
                result.append(pt)
        # Remove last if same as first (close() will handle closure)
        if len(result) > 2:
            dx = result[-1][0] - result[0][0]
            dy = result[-1][1] - result[0][1]
            if (dx * dx + dy * dy) <= tol * tol:
                result = result[:-1]
        return result

    def _loft_blade(self, sections) -> cq.Workplane:
        """Loft blade from airfoil sections at radial stations.

        Creates CadQuery wires on YZ workplanes offset along X (radial
        direction), then lofts through all wires to form a twisted blade.
        """
        # Pre-process all sections: center, rotate, collect 2D points
        wire_data = []
        for r, profile, twist in sections:
            twist_rad = math.radians(twist)
            cos_t = math.cos(twist_rad)
            sin_t = math.sin(twist_rad)

            # Center airfoil at mid-chord for minimum radial overshoot
            chord = np.max(profile[:, 0]) - np.min(profile[:, 0])
            centered = profile.copy()
            centered[:, 0] -= chord * 0.5

            # Rotate 2D profile by twist angle
            rotated_y = centered[:, 0] * cos_t - centered[:, 1] * sin_t
            rotated_z = centered[:, 0] * sin_t + centered[:, 1] * cos_t

            # Build 2D points (skip closing point — close() will handle it)
            pts = [(float(rotated_y[k]), float(rotated_z[k]))
                   for k in range(len(rotated_y) - 1)]
            pts = self._dedup_points(pts)
            wire_data.append((r, pts))

        # Build CadQuery loft through all sections
        try:
            result = cq.Workplane("YZ")
            for i, (r, pts) in enumerate(wire_data):
                if i == 0:
                    result = result.workplane(offset=r)
                else:
                    dr = r - wire_data[i - 1][0]
                    result = result.workplane(offset=dr)

                result = result.moveTo(pts[0][0], pts[0][1])
                for pt in pts[1:]:
                    result = result.lineTo(pt[0], pt[1])
                result = result.close()

            result = result.loft(ruled=True)
            return result

        except Exception:
            # Fallback: extrude root section across full blade span
            span = sections[-1][0] - sections[0][0]
            r0, pts0 = wire_data[0]
            fallback = cq.Workplane("YZ").workplane(offset=r0)
            fallback = fallback.moveTo(pts0[0][0], pts0[0][1])
            for pt in pts0[1:]:
                fallback = fallback.lineTo(pt[0], pt[1])
            fallback = fallback.close().extrude(span)
            return fallback

    def _add_magnet_pockets(self, ring: cq.Workplane) -> cq.Workplane:
        """Subtract magnet pockets from the blade ring inner wall."""
        pocket_spec = self.coupling.magnet_pocket_specs(self.stage_index)
        n_pockets = pocket_spec["num_pockets"]
        pocket_d = pocket_spec["pocket_diameter"]
        pocket_depth = pocket_spec["pocket_depth"]
        # Place pockets at ring inner wall (coupling interface)
        coupling_r = self.ring_inner_r
        angular_spacing = pocket_spec["angular_spacing"]
        ring_height = self.derived.get("blade_axial_width", 25)

        for i in range(n_pockets):
            angle = math.radians(i * angular_spacing)
            cx = coupling_r * math.cos(angle)
            cy = coupling_r * math.sin(angle)

            # Pocket centered axially on the ring (ring is centered at Z=0)
            pocket = (
                cq.Workplane("XY")
                .workplane(offset=-pocket_depth / 2)
                .center(cx, cy)
                .circle(pocket_d / 2)
                .extrude(pocket_depth)
            )
            ring = ring.cut(pocket)

        return ring
