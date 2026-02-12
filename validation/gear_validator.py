"""Gear mesh validation — contact ratio, undercutting, profile interference, stress.

Validates gear geometry for printability and mechanical correctness.
"""

import math
from dataclasses import dataclass
from typing import List

from src.gear_generator import GearGenerator


@dataclass
class GearValidationResult:
    """Result of a gear validation check."""
    check_name: str
    gear_name: str
    value: float
    limit: float
    passed: bool
    detail: str = ""


class GearValidator:
    """Validates gear geometry and meshing conditions."""

    def __init__(self, config: dict):
        self.config = config
        self.gen = GearGenerator(config)
        self.specs = self.gen.compute_gear_specs()

    def validate_all(self) -> List[GearValidationResult]:
        """Run all gear validations."""
        results = []
        results.extend(self.check_contact_ratios())
        results.extend(self.check_undercutting())
        results.extend(self.check_backlash())
        results.extend(self.check_agma_stress())
        return results

    def check_contact_ratios(self) -> List[GearValidationResult]:
        """Check contact ratio >= 1.2 for all mesh pairs."""
        results = []
        min_cr = 1.2

        # Sun-Planet mesh
        cr_sp = self.gen.contact_ratio(self.specs["sun"], self.specs["planet"])
        results.append(GearValidationResult(
            check_name="contact_ratio",
            gear_name="sun-planet",
            value=cr_sp,
            limit=min_cr,
            passed=cr_sp >= min_cr,
            detail=f"Contact ratio = {cr_sp:.3f} (minimum {min_cr})"
        ))

        # Planet-Ring mesh
        cr_pr = self.gen.contact_ratio(self.specs["planet"], self.specs["ring"])
        results.append(GearValidationResult(
            check_name="contact_ratio",
            gear_name="planet-ring",
            value=cr_pr,
            limit=min_cr,
            passed=cr_pr >= min_cr,
            detail=f"Contact ratio = {cr_pr:.3f} (minimum {min_cr})"
        ))

        return results

    def check_undercutting(self) -> List[GearValidationResult]:
        """Check for undercutting risk on all external gears."""
        results = []
        min_teeth = self.gen.min_teeth_no_undercut()

        for name in ["sun", "planet"]:
            spec = self.specs[name]
            undercut = self.gen.check_undercutting(spec)
            results.append(GearValidationResult(
                check_name="undercutting",
                gear_name=name,
                value=float(spec.teeth),
                limit=float(min_teeth),
                passed=not undercut,
                detail=(
                    f"{spec.teeth} teeth vs minimum {min_teeth} for "
                    f"{spec.pressure_angle}° pressure angle"
                    + (" — UNDERCUT RISK, consider profile shift or 25° PA" if undercut else "")
                )
            ))

        return results

    def check_backlash(self) -> List[GearValidationResult]:
        """Check backlash is in acceptable range for FDM printing."""
        results = []
        backlash = self.config["gears"]["backlash_compensation"]
        module = self.config["gears"]["module"]

        # Acceptable range: 0.10-0.20mm for module 1.0
        min_bl = 0.10 * module
        max_bl = 0.20 * module

        results.append(GearValidationResult(
            check_name="backlash",
            gear_name="all",
            value=backlash,
            limit=max_bl,
            passed=min_bl <= backlash <= max_bl,
            detail=f"Backlash = {backlash:.2f}mm, range [{min_bl:.2f}, {max_bl:.2f}]mm for module {module}"
        ))

        return results

    def check_agma_stress(self) -> List[GearValidationResult]:
        """Check AGMA bending stress with geometry factor.

        In a planetary gearset, the total sun torque is shared among
        num_planets meshes. The tangential force at each mesh is
        F_t = T_motor / (r_sun × num_planets). Both sun and planet
        teeth see this same mesh force.
        """
        results = []
        uts = self.config["materials"]["mechanical"]["tensile_strength"]  # MPa
        module = self.config["gears"]["module"]
        face_width = self.config["gears"]["gear_width"]
        num_planets = self.config["gears"]["num_planets"]
        motor_torque = self.config["motor"]["stall_torque"]  # N-m

        # Mesh tangential force: shared among planets, referenced to sun pitch circle
        sun_spec = self.specs["sun"]
        r_sun = sun_spec.pitch_diameter / 2  # mm
        F_t = (motor_torque * 1000) / (r_sun * num_planets)  # N

        for name in ["sun", "planet"]:
            spec = self.specs[name]

            # AGMA geometry factor J (approximate)
            J = 0.25 + 0.003 * spec.teeth
            J = min(J, 0.45)

            # AGMA bending stress: σ = F_t / (m × b × J)
            sigma = F_t / (module * face_width * J)

            sf = uts / sigma if sigma > 0 else float("inf")

            results.append(GearValidationResult(
                check_name="agma_bending",
                gear_name=name,
                value=sigma,
                limit=uts / 2.0,  # SF=2.0 required
                passed=sigma <= uts / 2.0,
                detail=f"σ={sigma:.1f}MPa, J={J:.3f}, F_t={F_t:.1f}N, SF={sf:.2f}"
            ))

        return results

    def all_passed(self) -> bool:
        """Check if all gear validations passed."""
        return all(r.passed for r in self.validate_all())
