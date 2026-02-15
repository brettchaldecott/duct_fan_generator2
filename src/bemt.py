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
        """Solve BEMT for all configured blade stages.

        Uses momentum-theory disk velocity as the reference axial velocity
        for static thrust conditions, with an outer iteration to refine
        v_disk from the computed total thrust.
        """
        # --- Estimate reference axial velocity from motor power ---
        motor = self.config["motor"]
        stall_torque = motor["stall_torque"]        # N-m
        no_load_rpm = motor["no_load_rpm"]           # RPM
        motor_rpm = motor["rpm"]                     # RPM (operating point)
        omega_motor = motor_rpm * 2 * math.pi / 60

        # Linear torque-speed model: T = T_stall * (1 - rpm/rpm_nl)
        torque_at_rpm = stall_torque * (1 - motor_rpm / no_load_rpm)
        motor_power = torque_at_rpm * omega_motor    # W

        # Disk area from blade radii (use first stage as representative)
        r_hub = self.derived["blade_hub_radius"] / 1000   # mm -> m
        r_tip = self.derived["blade_tip_radius"] / 1000
        disk_area = math.pi * (r_tip**2 - r_hub**2)

        # v_disk from momentum theory: P = 2*rho*A*v^3
        eta = 0.65  # overall efficiency (motor + transmission losses)
        v_disk = (motor_power * eta / (2 * self.RHO * disk_area)) ** (1/3)
        v_disk = max(v_disk, 1.0)  # floor at 1 m/s

        # --- Solve all stages with this v_disk ---
        stages = []
        inlet_swirl = None

        for i, blade_cfg in enumerate(self.config["blades"]["stages"]):
            stage_result = self.solve_stage(
                stage_index=i,
                blade_cfg=blade_cfg,
                rpm=self.derived["stage_rpms"][i],
                direction=self.derived["stage_directions"][i],
                inlet_swirl=inlet_swirl,
                v_axial_ref=v_disk,
            )
            stages.append(stage_result)
            inlet_swirl = (stage_result.exit_swirl, stage_result.radii)

        return BEMTResults(stages=stages)

    def solve_stage(self, stage_index: int, blade_cfg: dict, rpm: float,
                    direction: int, inlet_swirl=None,
                    v_axial_ref: float = 10.0) -> StageResult:
        """Solve BEMT for a single stage.

        Args:
            stage_index: 0-based stage index
            blade_cfg: Blade configuration dict
            rpm: Stage RPM
            direction: +1 (CW) or -1 (CCW)
            inlet_swirl: Optional (swirl_velocities, radii) from previous stage
            v_axial_ref: Reference axial velocity from disk loading (m/s)
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
                v_axial_ref=v_axial_ref,
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
                       v_inf=0, v_axial_ref=10.0,
                       max_iter=100, tol=1e-6) -> BladeSection:
        """Solve BEM equations for a single blade section.

        For static thrust (v_inf=0), uses the disk-loading reference velocity
        v_axial_ref instead of induction-based v_axial. This gives physically
        correct twist distribution: slow blades get steep pitch, fast blades
        get shallow pitch.
        """
        # Local blade speed
        u_theta = omega * r  # tangential velocity (m/s)

        # Initial guess for tangential induction
        a_prime = 0.01

        # Duct-aware max chord limits
        duct_ir = self.config["duct"]["inner_diameter"] / 2 / 1000  # mm -> m
        tip_cl_m = self.config["blades"]["tip_clearance"] / 1000     # mm -> m
        max_r = duct_ir - tip_cl_m
        max_chord_radial = 2 * math.sqrt(max(max_r**2 - r**2, 0))

        for iteration in range(max_iter):
            if v_inf > 0:
                # --- Forward flight path (preserved) ---
                a = 0.1 if iteration == 0 else a
                v_axial = v_inf * (1 + a)
            else:
                # --- Static thrust: use disk-loading velocity ---
                v_axial = v_axial_ref

            # Tangential velocity relative to blade
            v_tan = u_theta * (1 - a_prime) - direction * v_swirl_in

            # Flow angle
            phi = math.atan2(v_axial, v_tan) if v_tan > 0 else math.pi / 2
            # Clamp to valid range
            phi = max(0.01, min(phi, math.pi / 2))

            # Resultant velocity
            w = math.sqrt(v_axial**2 + v_tan**2)

            # Prandtl tip/hub loss
            f_tip = self._prandtl_factor(r, r_tip, n_blades, phi, mode="tip")
            f_hub = self._prandtl_factor(r, r_hub, n_blades, phi, mode="hub")
            F = max(f_tip * f_hub, 0.01)

            # Design angle of attack from thin airfoil theory
            alpha = design_cl / (2 * math.pi)
            twist = phi - alpha

            # Chord from Schmitz method (static thrust form)
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            chord = (8 * math.pi * r * sin_phi * cos_phi * F) / (n_blades * design_cl)
            chord = max(chord, 0.005)  # minimum 5mm

            # Apply chord limits
            chord = min(chord, max_chord_radial)
            chord = min(chord, 0.08)  # absolute maximum 80mm

            # Local solidity
            sigma = n_blades * chord / (2 * math.pi * r)

            # Cl/Cd model
            cl = design_cl
            cd = 0.008 + 0.01 * alpha**2

            # Force coefficients
            cn = cl * cos_phi + cd * sin_phi
            ct = cl * sin_phi - cd * cos_phi

            # Stable tangential induction formula:
            #   a' = sigma*ct / (4F*sin*cos + sigma*ct)
            # No subtraction in denominator — always well-behaved
            denom_ap = 4 * F * sin_phi * cos_phi + sigma * ct
            a_prime_new = (sigma * ct) / denom_ap if denom_ap > 0.001 else 0.01
            a_prime_new = max(0.0, min(a_prime_new, 0.25))

            if v_inf > 0:
                # Forward flight: also iterate on axial induction
                denom_a = 4 * F * sin_phi**2 / (sigma * cn) + 1
                a_new = 1.0 / denom_a if denom_a > 0 else 0.5
                if a_new > 0.4:
                    ac = 0.2
                    K = 4 * F * sin_phi**2 / (sigma * cn) if (sigma * cn) != 0 else 1e6
                    discriminant = (K * (1 - 2 * ac) + 2)**2 + 4 * (K * ac**2 - 1)
                    if discriminant < 0:
                        a_new = 0.4
                    else:
                        a_new = 0.5 * (2 + K * (1 - 2 * ac) - math.sqrt(discriminant))
                    a_new = max(0.0, min(a_new, 0.95))
            else:
                a_new = a_prime  # placeholder, will derive post-hoc

            # Convergence check (on a_prime for static, both for forward)
            if abs(a_prime_new - a_prime) < tol:
                if v_inf > 0 and abs(a_new - a) > tol:
                    pass  # keep iterating
                else:
                    a_prime = a_prime_new
                    if v_inf > 0:
                        a = a_new
                    break

            # Relaxation
            a_prime = 0.5 * a_prime + 0.5 * a_prime_new
            if v_inf > 0:
                a = 0.5 * a + 0.5 * a_new

        # --- Post-convergence: compute loads and derive a ---
        if v_inf <= 0:
            v_axial = v_axial_ref
        else:
            v_axial = v_inf * (1 + a)

        v_tan = u_theta * (1 - a_prime) - direction * v_swirl_in
        w = math.sqrt(v_axial**2 + v_tan**2)

        phi_eff = max(0.01, min(phi, math.pi / 2))
        sin_phi = math.sin(phi_eff)
        cos_phi = math.cos(phi_eff)
        cn_eff = cl * cos_phi + cd * sin_phi
        ct_eff = cl * sin_phi - cd * cos_phi

        # Blade element thrust and torque (per unit span, all blades)
        d_thrust = 0.5 * self.RHO * w**2 * chord * cn_eff * n_blades
        d_torque = 0.5 * self.RHO * w**2 * chord * ct_eff * n_blades * r

        # Ensure positive thrust
        d_thrust = abs(d_thrust)
        d_torque = abs(d_torque)

        # Derive axial induction post-hoc for reporting (static case)
        if v_inf <= 0:
            # From momentum: dT = 4*pi*r*rho*v_axial^2*a*F*dr
            # a = dT / (4*pi*r*rho*v_axial^2*F) (per unit span approximation)
            if v_axial_ref > 0 and F > 0.01:
                a = d_thrust / (4 * math.pi * r * self.RHO * v_axial_ref**2 * F)
                a = min(a, 0.95)
            else:
                a = 0.1

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
