"""Planetary gearbox generation.

Generates involute gear profiles for sun, planet, and ring gears.
Uses analytical involute tooth profile generation.
Supports backlash compensation for FDM printing.
Includes 3D solid generation via CadQuery for STL export.
"""

import math
import numpy as np
import cadquery as cq
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GearSpec:
    """Specification for a single gear."""
    name: str
    teeth: int
    module: float          # mm
    pressure_angle: float  # degrees
    pitch_diameter: float  # mm
    base_diameter: float   # mm
    outer_diameter: float  # mm
    root_diameter: float   # mm
    tooth_thickness: float # mm (at pitch circle)
    gear_width: float      # mm (face width)
    is_internal: bool = False


class GearGenerator:
    """Generates planetary gear system geometry."""

    def __init__(self, config: dict):
        self.config = config
        self.gears = config["gears"]
        self.module = self.gears["module"]
        self.pressure_angle = math.radians(self.gears["pressure_angle"])
        self.backlash = self.gears["backlash_compensation"]
        self.gear_width = self.gears["gear_width"]

    def compute_gear_specs(self) -> dict:
        """Compute gear specifications for all gears in the planetary set."""
        sun_teeth = self.gears["sun_teeth"]
        planet_teeth = self.gears["planet_teeth"]
        ring_teeth = self.gears["ring_teeth"]

        specs = {}

        # Sun gear
        specs["sun"] = self._make_spec("sun", sun_teeth, internal=False)

        # Planet gear
        specs["planet"] = self._make_spec("planet", planet_teeth, internal=False)

        # Ring gear (internal)
        specs["ring"] = self._make_spec("ring", ring_teeth, internal=True)

        return specs

    def _make_spec(self, name: str, teeth: int, internal: bool) -> GearSpec:
        """Create gear specification."""
        m = self.module
        pa = self.pressure_angle

        pitch_d = m * teeth
        base_d = pitch_d * math.cos(pa)

        if internal:
            outer_d = pitch_d - 2 * m  # internal: smaller
            root_d = pitch_d + 2 * 1.25 * m
        else:
            outer_d = pitch_d + 2 * m  # addendum = 1 module
            root_d = pitch_d - 2 * 1.25 * m  # dedendum = 1.25 module

        # Tooth thickness at pitch circle (with backlash compensation)
        tooth_thick = (math.pi * m / 2) - self.backlash / 2

        return GearSpec(
            name=name,
            teeth=teeth,
            module=m,
            pressure_angle=math.degrees(pa),
            pitch_diameter=pitch_d,
            base_diameter=base_d,
            outer_diameter=outer_d,
            root_diameter=root_d,
            tooth_thickness=tooth_thick,
            gear_width=self.gear_width,
            is_internal=internal,
        )

    def contact_ratio(self, gear_a: GearSpec, gear_b: GearSpec) -> float:
        """Calculate contact ratio between two meshing gears.

        Contact ratio = length of action / base pitch
        """
        pa = math.radians(gear_a.pressure_angle)
        m = gear_a.module

        # Base pitch
        pb = math.pi * m * math.cos(pa)

        # Radii
        if gear_b.is_internal:
            # External-internal mesh (planet meshing with ring)
            r_a = gear_a.pitch_diameter / 2   # planet pitch radius
            r_b = gear_b.pitch_diameter / 2   # ring pitch radius
            ra_a = gear_a.outer_diameter / 2  # planet tip radius
            # For internal gear, the "tip" is the inner surface
            # root_diameter for internal gear is the outer (larger) diameter
            # outer_diameter for internal gear is the inner (smaller) diameter
            ra_b = gear_b.outer_diameter / 2  # ring inner tip radius
            rb_a = gear_a.base_diameter / 2   # planet base radius
            rb_b = gear_b.base_diameter / 2   # ring base radius

            # Center distance for internal mesh
            C = r_b - r_a

            # Length of action for internal mesh:
            # Z = sqrt(ra_b² - rb_b²) - sqrt(ra_a² - rb_a²) + C × sin(φ)
            # But for internal gears this can also be computed as:
            term1 = math.sqrt(max(ra_a**2 - rb_a**2, 0))
            term2 = math.sqrt(max(rb_b**2 - ra_b**2, 0)) if rb_b > ra_b else \
                    math.sqrt(max(ra_b**2 - rb_b**2, 0))
            term3 = C * math.sin(pa)

            # For internal gear mesh: length = term1 + term3 - term2
            # or equivalently: sqrt(ra_pinion² - rb_pinion²) + C·sin(φ) - sqrt(ra_gear² - rb_gear²)
            length_of_action = abs(term1 + term3 - term2)
        else:
            # External-external mesh
            r_a = gear_a.pitch_diameter / 2
            r_b = gear_b.pitch_diameter / 2
            ra_a = gear_a.outer_diameter / 2
            ra_b = gear_b.outer_diameter / 2
            rb_a = gear_a.base_diameter / 2
            rb_b = gear_b.base_diameter / 2

            C = r_a + r_b

            term1 = math.sqrt(max(ra_a**2 - rb_a**2, 0))
            term2 = math.sqrt(max(ra_b**2 - rb_b**2, 0))
            term3 = C * math.sin(pa)

            length_of_action = term1 + term2 - term3

        cr = length_of_action / pb
        return cr

    def min_teeth_no_undercut(self) -> int:
        """Minimum teeth to avoid undercutting at the configured pressure angle."""
        pa = math.radians(self.gears["pressure_angle"])
        return math.ceil(2 / (math.sin(pa) ** 2))

    def check_undercutting(self, spec: GearSpec) -> bool:
        """Check if a gear will experience undercutting.

        Returns True if undercutting WILL occur (bad).
        """
        min_teeth = self.min_teeth_no_undercut()
        return spec.teeth < min_teeth and not spec.is_internal

    def gear_ratio(self) -> float:
        """Compute carrier-fixed planetary gear ratio (|R/S|)."""
        return self.gears["ring_teeth"] / self.gears["sun_teeth"]

    def planet_positions(self) -> List[float]:
        """Compute angular positions of planet gears (radians)."""
        n = self.gears["num_planets"]
        return [2 * math.pi * i / n for i in range(n)]

    def carrier_radius(self) -> float:
        """Center distance from sun to planet (carrier arm length)."""
        return (self.gears["sun_teeth"] + self.gears["planet_teeth"]) * self.module / 2

    def generate_involute_profile(self, spec: GearSpec, num_points: int = 50) -> np.ndarray:
        """Generate 2D involute tooth profile for a single tooth.

        Args:
            spec: Gear specification
            num_points: Points per involute curve

        Returns:
            numpy array of (x, y) coordinates for one tooth profile
        """
        rb = spec.base_diameter / 2  # base radius
        ra = spec.outer_diameter / 2 if not spec.is_internal else spec.root_diameter / 2
        rr = spec.root_diameter / 2 if not spec.is_internal else spec.outer_diameter / 2

        # Involute function: inv(α) = tan(α) - α
        def involute(alpha):
            return math.tan(alpha) - alpha

        # Generate involute curve points
        pa = math.radians(spec.pressure_angle)

        # Parameter range for involute
        if spec.is_internal:
            # Internal gear: involute goes inward
            alpha_max = math.acos(rb / ra) if rb < ra else 0
        else:
            alpha_max = math.acos(rb / ra) if rb < ra else 0

        alphas = np.linspace(0, alpha_max, num_points)

        # Involute curve (right flank)
        points = []
        for alpha in alphas:
            r = rb / math.cos(alpha) if math.cos(alpha) > 0.001 else rb
            theta = involute(alpha) - involute(pa)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            points.append([x, y])

        return np.array(points)

    def generate_full_gear_profile(self, spec: GearSpec, num_points: int = 50) -> np.ndarray:
        """Generate complete 2D gear profile (all teeth).

        Args:
            spec: Gear specification
            num_points: Points per involute curve

        Returns:
            numpy array of (x, y) coordinates for the complete gear profile
        """
        tooth_profile = self.generate_involute_profile(spec, num_points)
        tooth_angle = 2 * math.pi / spec.teeth

        all_points = []
        for i in range(spec.teeth):
            angle = i * tooth_angle
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Rotate tooth profile
            rotated = np.column_stack([
                tooth_profile[:, 0] * cos_a - tooth_profile[:, 1] * sin_a,
                tooth_profile[:, 0] * sin_a + tooth_profile[:, 1] * cos_a,
            ])

            # Mirror for other flank
            mirror_angle = angle + tooth_angle / 2
            cos_m = math.cos(mirror_angle)
            sin_m = math.sin(mirror_angle)
            mirrored = np.column_stack([
                tooth_profile[:, 0] * cos_m + tooth_profile[:, 1] * sin_m,
                tooth_profile[:, 0] * sin_m - tooth_profile[:, 1] * cos_m,
            ])

            all_points.extend(rotated.tolist())
            all_points.extend(mirrored[::-1].tolist())

        return np.array(all_points)

    @staticmethod
    def _dedup_points(points: List[Tuple[float, float]], tol: float = 1e-4) -> List[Tuple[float, float]]:
        """Remove consecutive duplicate points within tolerance."""
        if not points:
            return points
        result = [points[0]]
        for pt in points[1:]:
            dx = pt[0] - result[-1][0]
            dy = pt[1] - result[-1][1]
            if (dx * dx + dy * dy) > tol * tol:
                result.append(pt)
        # Remove last point if same as first
        if len(result) > 1:
            dx = result[-1][0] - result[0][0]
            dy = result[-1][1] - result[0][1]
            if (dx * dx + dy * dy) <= tol * tol:
                result = result[:-1]
        return result

    def _single_tooth_profile(self, spec: GearSpec, num_points: int = 15) -> List[Tuple[float, float]]:
        """Generate 2D profile for a single tooth centered at angle 0.

        The profile goes from root on the right flank, up the right involute,
        across the tip arc, down the left involute, to root on the left flank.
        This creates a closed shape suitable for extrusion and boolean union
        with a root circle disk.

        Args:
            spec: Gear specification (external only)
            num_points: Points per involute curve

        Returns:
            List of (x,y) tuples for the tooth profile
        """
        rb = spec.base_diameter / 2
        rr = spec.root_diameter / 2
        ra = spec.outer_diameter / 2
        pa = math.radians(spec.pressure_angle)

        def inv(alpha):
            return math.tan(alpha) - alpha

        # Half-tooth angular width at base circle
        half_thick = spec.tooth_thickness / (2 * (spec.pitch_diameter / 2)) + inv(pa)
        alpha_tip = math.acos(rb / ra) if rb < ra else 0

        pts = []
        alphas = np.linspace(0, alpha_tip, num_points)

        # Right involute start angle (at base circle)
        right_start_theta = half_thick

        # Start at root on right side (radial line from root to base if needed)
        pts.append((rr * math.cos(right_start_theta), rr * math.sin(right_start_theta)))
        if rb > rr:
            pts.append((rb * math.cos(right_start_theta), rb * math.sin(right_start_theta)))

        # Right involute: base to tip
        for a in alphas:
            r = rb / math.cos(a)
            theta = half_thick - inv(a) + inv(pa)
            pts.append((r * math.cos(theta), r * math.sin(theta)))

        # Tip arc: right to left
        right_tip_theta = half_thick - inv(alpha_tip) + inv(pa)
        left_tip_theta = -(half_thick - inv(alpha_tip) + inv(pa))
        for theta in np.linspace(right_tip_theta, left_tip_theta, 5)[1:-1]:
            pts.append((ra * math.cos(theta), ra * math.sin(theta)))

        # Left involute: tip to base (reversed)
        for a in reversed(alphas):
            r = rb / math.cos(a)
            theta = -(half_thick - inv(a) + inv(pa))
            pts.append((r * math.cos(theta), r * math.sin(theta)))

        # Base to root on left side (radial line if needed)
        left_start_theta = -half_thick
        if rb > rr:
            pts.append((rb * math.cos(left_start_theta), rb * math.sin(left_start_theta)))
        pts.append((rr * math.cos(left_start_theta), rr * math.sin(left_start_theta)))

        return self._dedup_points(pts)

    def _single_space_profile(self, spec: GearSpec, num_points: int = 15) -> List[Tuple[float, float]]:
        """Generate 2D profile for a single tooth space of a ring gear.

        The space profile represents the gap between internal teeth. It extends
        from inner_tip_r outward to outer_root_r, bounded by involute flanks.

        Args:
            spec: Ring gear specification (internal)
            num_points: Points per involute curve

        Returns:
            List of (x,y) tuples for the space profile
        """
        rb = spec.base_diameter / 2
        inner_tip_r = spec.outer_diameter / 2    # inner tip (smaller)
        outer_root_r = spec.root_diameter / 2    # outer root (larger)
        pa = math.radians(spec.pressure_angle)

        def inv(alpha):
            return math.tan(alpha) - alpha

        # Space angular half-width uses the complement of tooth thickness
        space_thick = math.pi * spec.module - spec.tooth_thickness
        space_half = space_thick / (2 * (spec.pitch_diameter / 2)) + inv(pa)

        alpha_max = math.acos(rb / outer_root_r) if rb < outer_root_r else 0
        alphas = np.linspace(0, alpha_max, num_points)

        # Build involute flanks from inner_tip_r to outer_root_r
        right_inv = []
        for a in alphas:
            r = rb / math.cos(a)
            if r < inner_tip_r:
                continue
            if r > outer_root_r:
                break
            theta = space_half - inv(a) + inv(pa)
            right_inv.append((r, theta))

        left_inv = []
        for a in alphas:
            r = rb / math.cos(a)
            if r < inner_tip_r:
                continue
            if r > outer_root_r:
                break
            theta = -(space_half - inv(a) + inv(pa))
            left_inv.append((r, theta))

        if not right_inv or not left_inv:
            return []

        pts = []
        # Inner tip arc (from left to right at inner_tip_r)
        l_inner_theta = left_inv[0][1]
        r_inner_theta = right_inv[0][1]
        for t in np.linspace(l_inner_theta, r_inner_theta, 5):
            pts.append((inner_tip_r * math.cos(t), inner_tip_r * math.sin(t)))

        # Right flank (inner to outer)
        for r, t in right_inv:
            pts.append((r * math.cos(t), r * math.sin(t)))

        # Outer root arc
        r_outer_theta = right_inv[-1][1]
        l_outer_theta = left_inv[-1][1]
        for t in np.linspace(r_outer_theta, l_outer_theta, 5)[1:-1]:
            pts.append((outer_root_r * math.cos(t), outer_root_r * math.sin(t)))

        # Left flank (outer to inner, reversed)
        for r, t in reversed(left_inv):
            pts.append((r * math.cos(t), r * math.sin(t)))

        return self._dedup_points(pts)

    def generate_gear_solid(self, spec: GearSpec, bore_diameter: float = 0) -> cq.Workplane:
        """Create a 3D CadQuery solid from a gear specification.

        External gears: root disk + boolean union of tooth solids.
        Internal (ring) gear: annular ring with tooth space slots cut out.

        Args:
            spec: Gear specification
            bore_diameter: Center bore diameter in mm (0 = no bore)

        Returns:
            CadQuery Workplane with the gear solid
        """
        if spec.is_internal:
            return self._generate_ring_gear_solid(spec)

        return self._generate_external_gear_solid(spec, bore_diameter)

    def _generate_external_gear_solid(self, spec: GearSpec, bore_diameter: float = 0) -> cq.Workplane:
        """Generate external gear solid using boolean tooth union.

        Creates root circle disk, then unions each tooth shape to it.

        Args:
            spec: Gear specification (external)
            bore_diameter: Center bore diameter in mm

        Returns:
            CadQuery Workplane with the gear solid
        """
        root_r = spec.root_diameter / 2
        tooth_profile = self._single_tooth_profile(spec)

        # Start with root circle disk
        gear = cq.Workplane("XY").circle(root_r).extrude(spec.gear_width)

        # Create one tooth solid and union rotated copies
        tooth_solid = (
            cq.Workplane("XY")
            .polyline(tooth_profile)
            .close()
            .extrude(spec.gear_width)
        )

        tooth_angle = 360.0 / spec.teeth
        for i in range(spec.teeth):
            rotated = tooth_solid.rotate((0, 0, 0), (0, 0, 1), i * tooth_angle)
            gear = gear.union(rotated)

        # Cut center bore
        if bore_diameter > 0:
            gear = (
                gear
                .faces(">Z").workplane()
                .circle(bore_diameter / 2)
                .cutThruAll()
            )

        return gear

    def _generate_ring_gear_solid(self, spec: GearSpec) -> cq.Workplane:
        """Generate ring (internal) gear solid.

        Creates annular ring, then cuts tooth space slots from the inner surface.

        Args:
            spec: Ring gear specification (must be internal)

        Returns:
            CadQuery Workplane with the ring gear solid
        """
        inner_tip_r = spec.outer_diameter / 2      # inner tip radius
        outer_root_r = spec.root_diameter / 2       # outer root radius
        outer_wall_r = outer_root_r + 2 * spec.module  # wall beyond root

        # Create annular ring (outer wall to inner tip)
        ring = (
            cq.Workplane("XY")
            .circle(outer_wall_r)
            .circle(inner_tip_r)
            .extrude(spec.gear_width)
        )

        # Create one tooth space slot and cut rotated copies
        space_profile = self._single_space_profile(spec)
        if len(space_profile) < 3:
            return ring

        space_solid = (
            cq.Workplane("XY")
            .polyline(space_profile)
            .close()
            .extrude(spec.gear_width)
        )

        tooth_angle = 360.0 / spec.teeth
        for i in range(spec.teeth):
            rotated = space_solid.rotate((0, 0, 0), (0, 0, 1), i * tooth_angle)
            ring = ring.cut(rotated)

        return ring

    def generate_planetary_stage(self, stage_index: int) -> Dict[str, cq.Workplane]:
        """Generate all gears for one planetary gear stage.

        Args:
            stage_index: Gear stage index (0-based)

        Returns:
            Dict of gear_name -> CadQuery Workplane solid
        """
        specs = self.compute_gear_specs()
        carrier_r = self.carrier_radius()
        angles = self.planet_positions()

        motor_shaft_d = self.config["motor"]["shaft_diameter"]
        # Planet bore: sized for a pin (half the motor shaft)
        planet_bore_d = motor_shaft_d

        solids = {}

        # Sun gear at origin with motor shaft bore
        sun = self.generate_gear_solid(specs["sun"], bore_diameter=motor_shaft_d)
        solids[f"gear_sun_stage_{stage_index}"] = sun

        # Planet gears translated to carrier radius at evenly spaced angles
        for j, angle in enumerate(angles):
            planet = self.generate_gear_solid(specs["planet"], bore_diameter=planet_bore_d)
            # Translate planet to its carrier position
            cx = carrier_r * math.cos(angle)
            cy = carrier_r * math.sin(angle)
            planet = planet.translate((cx, cy, 0))
            solids[f"gear_planet_{j}_stage_{stage_index}"] = planet

        # Ring gear centered at origin
        ring = self.generate_gear_solid(specs["ring"])
        solids[f"gear_ring_stage_{stage_index}"] = ring

        return solids
