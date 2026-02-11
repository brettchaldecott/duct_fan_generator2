"""Magnetic coupling design, analysis, and geometry.

Implements K&J-style torque estimation for synchronous magnetic couplings
and optionally uses magpylib for detailed simulation. Generates magnet
pocket geometry specifications.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# Magnet grade properties (Br in Tesla)
MAGNET_GRADES = {
    "N35": {"Br": 1.17, "BHmax": 263},
    "N42": {"Br": 1.29, "BHmax": 334},
    "N48": {"Br": 1.37, "BHmax": 382},
    "N52": {"Br": 1.44, "BHmax": 414},
}

# Permeability of free space
MU_0 = 4 * math.pi * 1e-7  # T·m/A


@dataclass
class CouplingStageResult:
    """Analysis results for a single magnetic coupling stage."""
    stage_index: int
    peak_torque: float          # N-m (maximum available torque)
    required_torque: float      # N-m (from BEMT/system)
    safety_factor: float
    required_sf: float
    passed: bool
    slip_angle: float           # degrees (angle at peak torque)
    coupling_radius: float      # mm
    num_pole_pairs: int
    magnet_diameter: float      # mm
    magnet_thickness: float     # mm
    wall_thickness: float       # mm
    torque_curve: Optional[np.ndarray] = None  # (angle_deg, torque_Nm) pairs


class MagneticCoupling:
    """Magnetic coupling analysis and geometry generation."""

    def __init__(self, config: dict):
        self.config = config
        self.coupling_cfg = config["magnetic_coupling"]
        self.grade = MAGNET_GRADES.get(
            self.coupling_cfg["magnet_grade"], MAGNET_GRADES["N52"]
        )

    def estimate_peak_torque(self, stage_cfg: dict) -> float:
        """Estimate peak torque for a coupling stage using analytical model.

        Simplified torque model based on magnetic dipole interaction:
        T_peak = n_poles * F_tangential * R_coupling

        where F_tangential ≈ (Br² × A × t) / (2 × μ₀ × (gap + t_wall)²)
        is a rough estimate of tangential force per pole pair.

        Args:
            stage_cfg: Configuration for one coupling stage

        Returns:
            Peak torque in N-m
        """
        Br = self.grade["Br"]  # Tesla
        d_mag = stage_cfg["magnet_diameter"] / 1000  # mm -> m
        t_mag = stage_cfg["magnet_thickness"] / 1000
        n_poles = stage_cfg["num_pole_pairs"]
        r_coupling = stage_cfg["coupling_radius"] / 1000
        t_wall = self.coupling_cfg["wall_thickness"] / 1000

        # Magnet face area
        A = math.pi * (d_mag / 2) ** 2

        # Air gap = wall thickness (both sides: drive + driven)
        gap = 2 * t_wall

        # Simplified magnetic force per pole pair
        # Based on attraction force between two magnets across a gap:
        # F = (Br² × A) / (2 × μ₀) × (t / (gap + t))²
        # This is a simplified model; magpylib gives more accurate results
        force_factor = (Br**2 * A) / (2 * MU_0)
        gap_factor = (t_mag / (gap + t_mag)) ** 2
        F_per_pole = force_factor * gap_factor

        # Peak tangential force (at 90°/n_poles mechanical angle)
        # For synchronous coupling: F_tangential ≈ F_attraction × sin(θ_slip)
        # Peak occurs at sin = 1, so F_tang_peak ≈ F_per_pole
        F_tangential = F_per_pole * 0.5  # ~50% of attraction becomes tangential

        # Total torque = sum over all pole pairs × coupling radius
        # Factor 2 for both inner and outer magnet rings
        T_peak = n_poles * 2 * F_tangential * r_coupling

        return T_peak

    def slip_angle(self, num_pole_pairs: int) -> float:
        """Mechanical angle at which peak torque occurs.

        For a synchronous coupling: θ_slip = 90° / num_pole_pairs

        Returns angle in degrees.
        """
        return 90.0 / num_pole_pairs

    def torque_curve(self, stage_cfg: dict, num_points: int = 90) -> np.ndarray:
        """Generate torque vs angle curve for a coupling stage.

        The coupling follows a sinusoidal torque-angle relationship:
        T(θ) = T_peak × sin(n_poles × θ)

        Args:
            stage_cfg: Coupling stage config
            num_points: Number of angular points

        Returns:
            Array of shape (num_points, 2): (angle_deg, torque_Nm)
        """
        T_peak = self.estimate_peak_torque(stage_cfg)
        n_poles = stage_cfg["num_pole_pairs"]

        # One full electrical cycle
        max_angle = 180.0 / n_poles  # mechanical degrees for half cycle
        angles = np.linspace(0, max_angle, num_points)
        torques = T_peak * np.sin(np.radians(n_poles * angles))

        return np.column_stack([angles, torques])

    def analyze_all_stages(self, required_torques: List[float]) -> List[CouplingStageResult]:
        """Analyze all coupling stages against required torques.

        Args:
            required_torques: Required torque per stage in N-m (from BEMT)

        Returns:
            List of CouplingStageResult
        """
        results = []
        required_sf = self.coupling_cfg["required_safety_factor"]

        for i, stage_cfg in enumerate(self.coupling_cfg["stages"]):
            peak_t = self.estimate_peak_torque(stage_cfg)
            req_t = required_torques[i] if i < len(required_torques) else 0.001
            req_t = max(req_t, 0.001)  # avoid division by zero

            sf = peak_t / req_t
            theta_slip = self.slip_angle(stage_cfg["num_pole_pairs"])
            curve = self.torque_curve(stage_cfg)

            results.append(CouplingStageResult(
                stage_index=i,
                peak_torque=peak_t,
                required_torque=req_t,
                safety_factor=sf,
                required_sf=required_sf,
                passed=sf >= required_sf,
                slip_angle=theta_slip,
                coupling_radius=stage_cfg["coupling_radius"],
                num_pole_pairs=stage_cfg["num_pole_pairs"],
                magnet_diameter=stage_cfg["magnet_diameter"],
                magnet_thickness=stage_cfg["magnet_thickness"],
                wall_thickness=self.coupling_cfg["wall_thickness"],
                torque_curve=curve,
            ))

        return results

    def magnet_pocket_specs(self, stage_index: int) -> dict:
        """Get magnet pocket geometry for a coupling stage.

        Returns dict with dimensions for generating pockets in the hub
        and blade ring.
        """
        stage = self.coupling_cfg["stages"][stage_index]
        clearance = self.coupling_cfg["magnet_pocket_clearance"]

        return {
            "pocket_diameter": stage["magnet_diameter"] + 2 * clearance,
            "pocket_depth": stage["magnet_thickness"] + 0.5,  # 0.5mm extra for glue
            "magnet_diameter": stage["magnet_diameter"],
            "magnet_thickness": stage["magnet_thickness"],
            "num_pockets": stage["num_pole_pairs"] * 2,  # N and S alternating
            "coupling_radius": stage["coupling_radius"],
            "angular_spacing": 360.0 / (stage["num_pole_pairs"] * 2),
        }

    def wall_flux_warning(self) -> str:
        """Generate warning about PETG CF ferromagnetic properties.

        PETG CF may have carbon fibers that could interact with magnetic field.
        """
        return (
            "WARNING: Hub wall at coupling zones uses PETG CF which may contain "
            "carbon fibers that could slightly attenuate magnetic flux. "
            "Consider using plain PETG for coupling wall zones if torque margins "
            "are tight. Test with actual printed samples before final assembly."
        )
