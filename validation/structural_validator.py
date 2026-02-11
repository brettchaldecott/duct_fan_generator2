"""Structural validation — material stress checks.

Validates centrifugal stress on blade roots, gear tooth stresses,
and hub wall hoop stress against material limits with safety factors.
"""

import math
from dataclasses import dataclass
from typing import List


@dataclass
class StressResult:
    """Result of a stress check."""
    component: str
    stress_type: str
    actual_stress: float    # MPa
    allowable_stress: float  # MPa
    safety_factor: float
    required_sf: float
    passed: bool
    detail: str = ""


class StructuralValidator:
    """Validates structural integrity of all components."""

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]
        self.blade_mat = config["materials"]["blade"]
        self.mech_mat = config["materials"]["mechanical"]

    def validate_all(self, bemt_results=None) -> List[StressResult]:
        """Run all structural validations."""
        results = []
        results.extend(self.check_blade_centrifugal())
        results.extend(self.check_gear_tooth_bending())
        results.extend(self.check_hub_wall_hoop())
        return results

    def check_blade_centrifugal(self) -> List[StressResult]:
        """Check centrifugal stress on blade roots for each stage.

        σ = 0.5 × ρ × ω² × (r_tip² - r_hub²)
        """
        results = []
        rho = self.blade_mat["density"]  # kg/m³
        uts = self.blade_mat["tensile_strength"]  # MPa
        required_sf = 3.0

        r_hub = self.derived["blade_hub_radius"] / 1000  # mm to m
        r_tip = self.derived["blade_tip_radius"] / 1000

        for i, rpm in enumerate(self.derived["stage_rpms"]):
            omega = rpm * 2 * math.pi / 60
            # Centrifugal stress at blade root (simplified uniform blade)
            sigma = 0.5 * rho * omega**2 * (r_tip**2 - r_hub**2)
            sigma_mpa = sigma / 1e6  # Pa to MPa

            allowable = uts / required_sf
            sf_actual = uts / sigma_mpa if sigma_mpa > 0 else float("inf")

            results.append(StressResult(
                component=f"blade_stage_{i+1}",
                stress_type="centrifugal",
                actual_stress=sigma_mpa,
                allowable_stress=allowable,
                safety_factor=sf_actual,
                required_sf=required_sf,
                passed=sigma_mpa <= allowable,
                detail=f"RPM={rpm}, ω={omega:.1f} rad/s, r_hub={r_hub*1000:.1f}mm, r_tip={r_tip*1000:.1f}mm"
            ))

        return results

    def check_gear_tooth_bending(self) -> List[StressResult]:
        """Check gear tooth bending stress using Lewis equation.

        σ = F_t / (m × b × Y)
        where F_t = torque / (pitch_radius), m = module, b = face width,
        Y = Lewis form factor
        """
        results = []
        uts = self.mech_mat["tensile_strength"]  # MPa
        required_sf = 2.0

        gears = self.config["gears"]
        module = gears["module"]  # mm
        face_width = gears["gear_width"]  # mm
        sun_teeth = gears["sun_teeth"]
        planet_teeth = gears["planet_teeth"]

        # Lewis form factor approximation (for 20° pressure angle)
        def lewis_Y(z):
            return 0.154 - 0.912 / z

        # Motor torque at each gear stage
        motor_torque = self.config["motor"]["stall_torque"]  # N-m (worst case)

        for gear_name, teeth in [("sun", sun_teeth), ("planet", planet_teeth)]:
            Y = lewis_Y(teeth)
            if Y <= 0:
                Y = 0.01  # Undercut gear — flag separately

            pitch_radius = module * teeth / 2  # mm
            # Tangential force on tooth
            F_t = (motor_torque * 1000) / pitch_radius  # N (torque in N-mm / radius in mm)

            # Lewis bending stress
            sigma = F_t / (module * face_width * Y)  # MPa

            allowable = uts / required_sf
            sf_actual = uts / sigma if sigma > 0 else float("inf")

            results.append(StressResult(
                component=f"gear_{gear_name}",
                stress_type="tooth_bending_lewis",
                actual_stress=sigma,
                allowable_stress=allowable,
                safety_factor=sf_actual,
                required_sf=required_sf,
                passed=sigma <= allowable,
                detail=f"teeth={teeth}, Y={Y:.3f}, F_t={F_t:.1f}N"
            ))

        return results

    def check_hub_wall_hoop(self) -> List[StressResult]:
        """Check hub wall hoop stress at magnetic coupling zones.

        Simplified thin-wall hoop stress from magnetic pull force:
        σ_hoop = F_pull × r / (wall_thickness × length)
        """
        results = []
        uts = self.mech_mat["tensile_strength"]  # MPa
        required_sf = 3.0

        wall_t = self.config["magnetic_coupling"]["wall_thickness"]  # mm
        hub_od = self.derived["hub_od"]  # mm
        hub_r = hub_od / 2

        for i, coupling in enumerate(self.config["magnetic_coupling"]["stages"]):
            # Approximate magnetic pull force (simplified dipole attraction)
            # F ≈ B²A/(2μ₀) per pole pair, very rough estimate
            magnet_d = coupling["magnet_diameter"]
            magnet_t = coupling["magnet_thickness"]
            n_poles = coupling["num_pole_pairs"]
            # Rough pull force estimate: ~5N per pole pair for N52 8mm magnets
            f_pull_per_pole = 5.0 * (magnet_d / 8.0)**2 * (magnet_t / 3.0)
            f_pull_total = f_pull_per_pole * n_poles * 2  # both sides

            # Hoop stress: convert force to equivalent pressure on thin cylinder,
            # then apply σ_hoop = p × r / t.
            # p = F / (2π × r × L), so σ = F / (2π × t × L)
            coupling_zone_length = magnet_d + 4  # mm, zone length
            sigma_mpa = f_pull_total / (2 * math.pi * wall_t * coupling_zone_length)

            allowable = uts / required_sf
            sf_actual = uts / sigma_mpa if sigma_mpa > 0 else float("inf")

            results.append(StressResult(
                component=f"hub_wall_stage_{i+1}",
                stress_type="hoop_magnetic",
                actual_stress=sigma_mpa,
                allowable_stress=allowable,
                safety_factor=sf_actual,
                required_sf=required_sf,
                passed=sigma_mpa <= allowable,
                detail=f"F_pull={f_pull_total:.1f}N, wall={wall_t}mm, r={hub_r:.1f}mm"
            ))

        return results

    def all_passed(self, results: List[StressResult] = None) -> bool:
        """Check if all structural validations passed."""
        if results is None:
            results = self.validate_all()
        return all(r.passed for r in results)
