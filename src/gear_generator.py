"""Planetary gearbox generation.

Generates involute gear profiles for sun, planet, and ring gears.
Uses analytical involute tooth profile generation (cq_gears optional).
Supports backlash compensation for FDM printing.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


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
