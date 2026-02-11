"""Magnetic coupling validation — torque margin checks.

Validates that magnetic coupling provides sufficient torque margin
over required aerodynamic torque at all stages.
"""

from dataclasses import dataclass
from typing import List

from src.magnetic_coupling import MagneticCoupling, CouplingStageResult


@dataclass
class CouplingValidation:
    """Summary of coupling validation results."""
    stages: List[CouplingStageResult]
    all_passed: bool
    critical_stage: int  # index of stage with lowest margin
    ferromagnetic_warning: str


class CouplingValidator:
    """Validates magnetic coupling torque margins."""

    def __init__(self, config: dict):
        self.config = config
        self.coupling = MagneticCoupling(config)

    def validate(self, required_torques: List[float]) -> CouplingValidation:
        """Validate coupling torque margins for all stages.

        Args:
            required_torques: Required torque per stage in N-m (from BEMT)
        """
        results = self.coupling.analyze_all_stages(required_torques)

        # Find critical stage (lowest safety factor)
        critical_idx = min(range(len(results)), key=lambda i: results[i].safety_factor)

        return CouplingValidation(
            stages=results,
            all_passed=all(r.passed for r in results),
            critical_stage=critical_idx,
            ferromagnetic_warning=self.coupling.wall_flux_warning(),
        )

    def validate_with_bemt(self, bemt_results) -> CouplingValidation:
        """Validate using BEMT results directly."""
        required_torques = [stage.total_torque for stage in bemt_results.stages]
        return self.validate(required_torques)
