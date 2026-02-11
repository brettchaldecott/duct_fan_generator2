"""Blade Element Momentum Theory (BEMT) solver.

Iterative BEMT with Prandtl tip/hub loss factors, Glauert high-induction
correction, and duct augmentation factor. Supports multi-stage counter-rotating
configurations with swirl coupling between stages.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from src.airfoil import parse_naca4


@dataclass
class BladeSection:
    """Results for a single radial blade section."""
    radius: float          # m (from axis)
    chord: float           # m
    twist: float           # radians
    alpha: float           # angle of attack, radians
    cl: float
    cd: float
    d_thrust: float        # N (elemental thrust)
    d_torque: float        # N-m (elemental torque)
    a: float               # axial induction factor
    a_prime: float         # tangential induction factor
    phi: float             # flow angle, radians
    tip_loss: float        # Prandtl tip loss factor


@dataclass
class StageResult:
    """BEMT results for a single fan stage."""
    stage_index: int
    rpm: float
    direction: int              # +1 CW, -1 CCW
    sections: List[BladeSection]
    total_thrust: float         # N
    total_torque: float         # N-m
    power: float                # W
    exit_swirl: np.ndarray      # tangential velocity at each section (m/s)
    radii: np.ndarray           # radial positions (m)

    @property
    def efficiency(self) -> float:
        """Propulsive efficiency T*V / P (simplified)."""
        if self.power <= 0:
            return 0.0
        # Use mean axial velocity from induction
        mean_a = np.mean([s.a for s in self.sections])
        v_inf = 0  # static thrust case
        v_disk = v_inf * (1 + mean_a) if v_inf > 0 else 0
        return 0.0  # Static case - no meaningful propulsive efficiency


@dataclass
class BEMTResults:
    """Results for all stages combined."""
    stages: List[StageResult]

    @property
    def total_thrust(self) -> float:
        return sum(s.total_thrust for s in self.stages)

    @property
    def total_power(self) -> float:
        return sum(s.power for s in self.stages)


class BEMTSolver:
    """Blade Element Momentum Theory solver for ducted fan stages."""

    # Duct augmentation factor (ducted fans produce more thrust than open rotors)
    DUCT_AUGMENTATION = 1.2

    # Air properties at sea level
    RHO = 1.225  # kg/m³

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]

    def solve_all_stages(self) -> BEMTResults:
        """Solve BEMT for all configured blade stages."""
        stages = []
        inlet_swirl = None  # No inlet swirl for first stage

        for i, blade_cfg in enumerate(self.config["blades"]["stages"]):
            stage_result = self.solve_stage(
                stage_index=i,
                blade_cfg=blade_cfg,
                rpm=self.derived["stage_rpms"][i],
                direction=self.derived["stage_directions"][i],
                inlet_swirl=inlet_swirl,
            )
            stages.append(stage_result)
            # Pass exit swirl to next stage as inlet swirl
            inlet_swirl = (stage_result.exit_swirl, stage_result.radii)

        return BEMTResults(stages=stages)

    def solve_stage(self, stage_index: int, blade_cfg: dict, rpm: float,
                    direction: int, inlet_swirl=None) -> StageResult:
        """Solve BEMT for a single stage.

        Args:
            stage_index: 0-based stage index
            blade_cfg: Blade configuration dict
            rpm: Stage RPM
            direction: +1 (CW) or -1 (CCW)
            inlet_swirl: Optional (swirl_velocities, radii) from previous stage
        """
        omega = rpm * 2 * math.pi / 60  # rad/s
        n_blades = blade_cfg["num_blades"]
        design_cl = blade_cfg["design_cl"]

        # Parse airfoil parameters for Cl/Cd model
        m_root, p_root, t_root = parse_naca4(blade_cfg["airfoil_root"])
        m_tip, p_tip, t_tip = parse_naca4(blade_cfg["airfoil_tip"])

        # Radial stations
        r_hub = self.derived["blade_hub_radius"] / 1000  # convert mm to m
        r_tip = self.derived["blade_tip_radius"] / 1000

        num_sections = self.config["blades"]["num_radial_sections"]
        radii = np.linspace(r_hub, r_tip, num_sections + 2)[1:-1]  # exclude hub/tip edges

        sections = []
        total_thrust = 0
        total_torque = 0
        exit_swirl = np.zeros(len(radii))

        for j, r in enumerate(radii):
            # Spanwise fraction
            frac = (r - r_hub) / (r_tip - r_hub)

            # Interpolated airfoil properties
            t_local = t_root * (1 - frac) + t_tip * frac
            m_local = m_root * (1 - frac) + m_tip * frac

            # Zero-lift angle (approximate for cambered airfoil)
            alpha_0 = -m_local * 10  # rough approximation, radians

            # Inlet swirl from previous stage
            v_swirl_in = 0
            if inlet_swirl is not None:
                swirl_vel, swirl_radii = inlet_swirl
                v_swirl_in = float(np.interp(r, swirl_radii, swirl_vel))

            # Solve section with BEM iteration
            section = self._solve_section(
                r=r,
                r_hub=r_hub,
                r_tip=r_tip,
                omega=omega,
                n_blades=n_blades,
                design_cl=design_cl,
                alpha_0=alpha_0,
                t_over_c=t_local,
                direction=direction,
                v_swirl_in=v_swirl_in,
            )
            sections.append(section)

            # Accumulate thrust and torque (trapezoidal integration)
            dr = (r_tip - r_hub) / (num_sections + 1)
            total_thrust += section.d_thrust * dr
            total_torque += section.d_torque * dr

            # Exit swirl velocity
            exit_swirl[j] = 2 * section.a_prime * omega * r * direction

        # Apply duct augmentation
        total_thrust *= self.DUCT_AUGMENTATION

        power = total_torque * omega

        return StageResult(
            stage_index=stage_index,
            rpm=rpm,
            direction=direction,
            sections=sections,
            total_thrust=total_thrust,
            total_torque=total_torque,
            power=power,
            exit_swirl=exit_swirl,
            radii=radii,
        )

    def _solve_section(self, r, r_hub, r_tip, omega, n_blades, design_cl,
                       alpha_0, t_over_c, direction, v_swirl_in,
                       v_inf=0, max_iter=100, tol=1e-6) -> BladeSection:
        """Solve BEM equations for a single blade section.

        Uses iterative approach to find axial (a) and tangential (a')
        induction factors.
        """
        # Local speed
        u_theta = omega * r  # tangential velocity (m/s)

        # Initial guesses
        a = 0.1       # axial induction
        a_prime = 0.01  # tangential induction

        # Solidity
        # Initial chord estimate from design Cl
        sigma_approx = 0.1  # initial guess

        converged = False
        for iteration in range(max_iter):
            # Axial velocity through disk
            v_axial = v_inf * (1 + a) if v_inf > 0 else max(a * u_theta, 0.1)

            # Tangential velocity relative to blade
            v_tan = u_theta * (1 - a_prime) + v_swirl_in

            # Flow angle
            phi = math.atan2(v_axial, v_tan) if v_tan != 0 else math.pi / 2

            # Resultant velocity
            w = math.sqrt(v_axial**2 + v_tan**2)

            # Prandtl tip loss factor
            f_tip = self._prandtl_factor(r, r_tip, n_blades, phi, mode="tip")
            f_hub = self._prandtl_factor(r, r_hub, n_blades, phi, mode="hub")
            F = f_tip * f_hub
            F = max(F, 0.01)  # prevent division by zero

            # Design: choose chord and twist to achieve design_cl at this phi
            alpha = design_cl / (2 * math.pi)  # thin airfoil theory: cl = 2*pi*alpha
            twist = phi - alpha

            # Chord from blade loading (Schmitz method)
            chord = (8 * math.pi * r * math.sin(phi) * F) / (n_blades * design_cl) * a / max(1 - a, 0.01)
            chord = max(chord, 0.005)  # minimum 5mm chord
            chord = min(chord, 0.08)   # maximum 80mm chord

            # Local solidity
            sigma = n_blades * chord / (2 * math.pi * r)

            # Cl/Cd model
            cl = design_cl
            cd = 0.008 + 0.01 * alpha**2  # simple quadratic drag polar

            # Normal and tangential force coefficients
            cn = cl * math.cos(phi) + cd * math.sin(phi)
            ct = cl * math.sin(phi) - cd * math.cos(phi)

            # New induction factors
            denom_a = 4 * F * math.sin(phi)**2 / (sigma * cn) + 1
            a_new = 1.0 / denom_a if denom_a > 0 else 0.5

            denom_ap = 4 * F * math.sin(phi) * math.cos(phi) / (sigma * ct) - 1
            a_prime_new = 1.0 / denom_ap if denom_ap > 0 else 0.01

            # Glauert correction for high induction (a > 0.4)
            if a_new > 0.4:
                # Buhl's correction
                ac = 0.2  # critical induction factor
                K = 4 * F * math.sin(phi)**2 / (sigma * cn) if (sigma * cn) != 0 else 1e6
                discriminant = (K * (1 - 2 * ac) + 2)**2 + 4 * (K * ac**2 - 1)
                if discriminant < 0:
                    # Fallback: cap induction at reasonable value
                    a_new = 0.4
                else:
                    a_new = 0.5 * (2 + K * (1 - 2 * ac) -
                                   math.sqrt(discriminant))
                a_new = max(0.0, min(a_new, 0.95))

            a_prime_new = max(0.0, min(a_prime_new, 0.5))

            # Check convergence
            if abs(a_new - a) < tol and abs(a_prime_new - a_prime) < tol:
                converged = True
                a = a_new
                a_prime = a_prime_new
                break

            # Relaxation
            a = 0.5 * a + 0.5 * a_new
            a_prime = 0.5 * a_prime + 0.5 * a_prime_new

        # Compute elemental loads using blade element theory
        v_axial = v_inf * (1 + a) if v_inf > 0 else max(a * u_theta, 0.1)
        v_tan = u_theta * (1 - a_prime) + v_swirl_in
        w = math.sqrt(v_axial**2 + v_tan**2)

        # Ensure phi is in valid range for thrust production
        # For a working fan, phi should be between 0 and pi/2
        phi_eff = max(0.01, min(abs(phi), math.pi / 2))
        cn_eff = cl * math.cos(phi_eff) + cd * math.sin(phi_eff)
        ct_eff = cl * math.sin(phi_eff) - cd * math.cos(phi_eff)

        # Blade element thrust and torque (per unit span, all blades)
        d_thrust = 0.5 * self.RHO * w**2 * chord * cn_eff * n_blades
        d_torque = 0.5 * self.RHO * w**2 * chord * ct_eff * n_blades * r

        # Ensure positive thrust (fan always pushes air)
        d_thrust = abs(d_thrust)
        d_torque = abs(d_torque)

        return BladeSection(
            radius=r,
            chord=chord,
            twist=twist,
            alpha=alpha,
            cl=cl,
            cd=cd,
            d_thrust=d_thrust,
            d_torque=d_torque,
            a=a,
            a_prime=a_prime,
            phi=phi,
            tip_loss=F,
        )

    @staticmethod
    def _prandtl_factor(r, r_edge, n_blades, phi, mode="tip"):
        """Prandtl tip/hub loss factor.

        Args:
            r: Current radial position (m)
            r_edge: Tip or hub radius (m)
            n_blades: Number of blades
            phi: Flow angle (radians)
            mode: 'tip' or 'hub'
        """
        sin_phi = abs(math.sin(phi))
        if sin_phi < 1e-6:
            return 1.0

        if mode == "tip":
            if r >= r_edge:
                return 0.0
            f = (n_blades / 2) * (r_edge - r) / (r * sin_phi)
        else:
            if r <= r_edge:
                return 0.0
            f = (n_blades / 2) * (r - r_edge) / (r * sin_phi)

        if f > 20:
            return 1.0

        F = (2 / math.pi) * math.acos(math.exp(-f))
        return F
