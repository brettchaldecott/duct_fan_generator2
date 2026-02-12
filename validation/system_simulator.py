"""Full power chain system simulation.

Simulates the complete power chain from motor through gears and magnetic
coupling to fan load. One shared motor drives all stages simultaneously
through the gearbox. Finds steady-state operating point and simulates
startup transients.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

from src.magnetic_coupling import MagneticCoupling


@dataclass
class OperatingPoint:
    """Steady-state operating point for a stage."""
    stage_index: int
    rpm: float
    motor_torque: float       # N-m
    aero_torque: float        # N-m
    coupling_torque_angle: float  # degrees
    coupling_slip_pct: float  # % of slip angle used
    power_in: float           # W (electrical)
    power_shaft: float        # W (after motor losses)
    power_after_gears: float  # W (after gear losses)
    power_thrust: float       # W (useful thrust power)
    efficiency: float         # overall


@dataclass
class TransientResult:
    """Transient simulation results."""
    time: np.ndarray          # seconds
    speed: np.ndarray         # RPM
    coupling_angle: np.ndarray  # degrees
    max_coupling_angle: float   # peak coupling angle during startup
    slip_angle: float           # coupling slip angle
    coupling_margin_pct: float  # min margin during transient


@dataclass
class SystemSimResult:
    """Complete system simulation results."""
    operating_points: list     # OperatingPoint per stage
    transient: Optional[TransientResult]
    power_budget: dict
    converged: bool


class SystemSimulator:
    """Full power chain steady-state and transient simulation."""

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]
        self.motor = config["motor"]
        self.gears = config["gears"]
        self.coupling = MagneticCoupling(config)

    def motor_torque(self, rpm: float) -> float:
        """Linear motor torque-speed model.

        T = T_stall * (1 - RPM / RPM_no_load)
        """
        t_stall = self.motor["stall_torque"]
        rpm_nl = self.motor["no_load_rpm"]
        if rpm >= rpm_nl:
            return 0.0
        return t_stall * (1 - rpm / rpm_nl)

    def motor_power(self, rpm: float) -> float:
        """Motor mechanical power output."""
        omega = rpm * 2 * math.pi / 60
        return self.motor_torque(rpm) * omega

    def gear_efficiency_for_stage(self, stage_index: int) -> float:
        """Gear train efficiency for a specific stage.

        Stage 0: direct drive (no gears) -> eta = 1.0
        Stage 1: 1 gear stage -> eta = 0.95
        Stage 2: 2 gear stages -> eta = 0.95^2
        """
        eta_per_stage = self.gears["efficiency_per_stage"]
        n_gear_stages = min(stage_index, self.gears["num_stages"])
        return eta_per_stage ** n_gear_stages

    def gear_ratio_for_stage(self, stage_index: int) -> float:
        """Total gear ratio from motor to a specific stage.

        Stage 0: ratio = 1.0 (direct drive)
        Stage 1: ratio = gear_ratio^1
        Stage 2: ratio = gear_ratio^2
        """
        gear_ratio = self.derived["gear_ratio"]
        n_gear_stages = min(stage_index, self.gears["num_stages"])
        return gear_ratio ** n_gear_stages

    def aero_torque_model(self, rpm: float, stage_index: int,
                          bemt_results=None) -> float:
        """Aerodynamic torque model.

        Uses quadratic scaling from BEMT reference point:
        Q(omega) = Q_ref * (omega / omega_ref)^2
        """
        if bemt_results is not None and stage_index < len(bemt_results.stages):
            ref = bemt_results.stages[stage_index]
            ref_rpm = ref.rpm
            ref_torque = ref.total_torque
            if ref_rpm > 0:
                return ref_torque * (rpm / ref_rpm) ** 2
        # Fallback: rough estimate
        omega = rpm * 2 * math.pi / 60
        return 1e-8 * omega ** 2

    def find_system_equilibrium(self, bemt_results=None) -> list:
        """Find shared-motor steady-state equilibrium for all stages.

        Single unknown: motor RPM (omega_m).
        All stage RPMs derived: omega_stage_i = omega_m / ratio_i
        Equilibrium: T_motor(omega_m) = sum[ T_aero_i(omega_m/R_i) / (R_i * eta_i) ]

        Returns list of OperatingPoint, one per stage.
        """
        num_stages = len(self.config["blades"]["stages"])

        def total_reflected_load(motor_rpm):
            """Total aero torque reflected to motor shaft."""
            total = 0.0
            for i in range(num_stages):
                ratio = self.gear_ratio_for_stage(i)
                eta = self.gear_efficiency_for_stage(i)
                fan_rpm = motor_rpm / ratio
                t_aero = self.aero_torque_model(fan_rpm, i, bemt_results)
                # Reflect aero torque to motor shaft: T_motor = T_aero / (ratio * eta)
                if ratio > 0 and eta > 0:
                    total += t_aero / (ratio * eta)
            return total

        def torque_balance(motor_rpm):
            """Residual: motor torque - total reflected load."""
            if motor_rpm >= self.motor["no_load_rpm"]:
                return -1.0
            t_motor = self.motor_torque(motor_rpm)
            t_load = total_reflected_load(motor_rpm)
            return t_motor - t_load

        # Search for equilibrium in motor RPM space
        rpm_min = 1.0
        rpm_max = self.motor["no_load_rpm"] - 1.0

        try:
            f_min = torque_balance(rpm_min)
            f_max = torque_balance(rpm_max)

            if f_min * f_max > 0:
                # No sign change — use design motor RPM as approximation
                eq_motor_rpm = float(self.motor["rpm"])
            else:
                eq_motor_rpm = brentq(torque_balance, rpm_min, rpm_max, xtol=1.0)
        except (ValueError, RuntimeError):
            eq_motor_rpm = float(self.motor["rpm"])

        # Build operating points for each stage at equilibrium
        t_motor = self.motor_torque(eq_motor_rpm)
        omega_motor = eq_motor_rpm * 2 * math.pi / 60
        p_shaft = t_motor * omega_motor
        motor_eff = 0.85
        p_electrical = p_shaft / motor_eff if motor_eff > 0 else p_shaft

        operating_points = []
        for i in range(num_stages):
            ratio = self.gear_ratio_for_stage(i)
            eta = self.gear_efficiency_for_stage(i)
            fan_rpm = eq_motor_rpm / ratio
            t_aero = self.aero_torque_model(fan_rpm, i, bemt_results)

            # Coupling analysis
            if i < len(self.config["magnetic_coupling"]["stages"]):
                stage_cfg = self.config["magnetic_coupling"]["stages"][i]
                peak_torque = self.coupling.estimate_peak_torque(stage_cfg)
                slip_angle = self.coupling.slip_angle(stage_cfg["num_pole_pairs"])

                if peak_torque > 0:
                    sin_val = min(t_aero / peak_torque, 1.0)
                    n_poles = stage_cfg["num_pole_pairs"]
                    coupling_angle = math.degrees(math.asin(sin_val)) / n_poles
                    coupling_slip_pct = coupling_angle / slip_angle * 100
                else:
                    coupling_angle = 0
                    coupling_slip_pct = 0
            else:
                coupling_angle = 0
                coupling_slip_pct = 0

            # Power for this stage
            omega_fan = fan_rpm * 2 * math.pi / 60
            p_after_gears = p_shaft * eta / num_stages  # share of shaft power after gear losses
            p_thrust = t_aero * omega_fan

            # Per-stage share of electrical input
            p_in_stage = p_electrical / num_stages
            stage_efficiency = p_thrust / p_in_stage if p_in_stage > 0 else 0

            operating_points.append(OperatingPoint(
                stage_index=i,
                rpm=fan_rpm,
                motor_torque=t_motor,
                aero_torque=t_aero,
                coupling_torque_angle=coupling_angle,
                coupling_slip_pct=coupling_slip_pct,
                power_in=p_in_stage,
                power_shaft=p_shaft / num_stages,
                power_after_gears=p_after_gears,
                power_thrust=p_thrust,
                efficiency=stage_efficiency,
            ))

        return operating_points

    def simulate_startup(self, stage_index: int = 0,
                         bemt_results=None,
                         t_final: float = 2.0) -> TransientResult:
        """Simulate startup transient to check coupling doesn't slip.

        Models shared motor driving all stages. Single DOF: motor RPM.
        J_total * d(omega)/dt = T_motor - sum(T_load_reflected)
        """
        num_stages = len(self.config["blades"]["stages"])

        # Total reflected inertia from all stages
        r_tip = self.derived["blade_tip_radius"] / 1000
        r_hub = self.derived["blade_hub_radius"] / 1000
        m_ring = 0.05  # kg per stage (rough blade ring mass)

        J_total = 0.0
        for i in range(num_stages):
            ratio = self.gear_ratio_for_stage(i)
            J_stage = 0.5 * m_ring * (r_tip**2 + r_hub**2)
            # Reflect inertia to motor shaft: J_reflected = J / ratio^2
            J_total += J_stage / (ratio**2)

        # Add motor rotor inertia estimate
        J_motor = 1e-4  # kg*m^2 (typical small BLDC)
        J_total += J_motor

        # Coupling parameters for the primary stage (for margin tracking)
        if stage_index < len(self.config["magnetic_coupling"]["stages"]):
            stage_cfg = self.config["magnetic_coupling"]["stages"][stage_index]
            peak_torque = self.coupling.estimate_peak_torque(stage_cfg)
            n_poles = stage_cfg["num_pole_pairs"]
            s_angle = self.coupling.slip_angle(n_poles)
        else:
            peak_torque = 1.0
            n_poles = 6
            s_angle = 15.0

        def dynamics(t, state):
            omega_motor = state[0]  # rad/s (motor shaft)
            motor_rpm = omega_motor * 60 / (2 * math.pi)
            motor_rpm = max(0, min(motor_rpm, self.motor["no_load_rpm"] - 1))

            t_motor = self.motor_torque(motor_rpm)

            # Total reflected load from all stages
            t_load_total = 0.0
            for i in range(num_stages):
                ratio = self.gear_ratio_for_stage(i)
                eta = self.gear_efficiency_for_stage(i)
                fan_rpm = motor_rpm / ratio
                t_aero = self.aero_torque_model(max(fan_rpm, 0), i, bemt_results)
                if ratio > 0 and eta > 0:
                    t_load_total += t_aero / (ratio * eta)

            # Limit drive to coupling capacity
            net_torque = min(t_motor, peak_torque) - t_load_total
            d_omega = net_torque / J_total
            return [d_omega]

        sol = solve_ivp(
            dynamics,
            [0, t_final],
            [0.0],
            max_step=0.01,
            method='RK45',
        )

        # Convert motor shaft speed to stage 0 fan speed for output
        ratio_0 = self.gear_ratio_for_stage(stage_index)
        speed_rpm = sol.y[0] * 60 / (2 * math.pi) / ratio_0

        # Estimate coupling angle during transient
        coupling_angles = []
        for rpm_val in speed_rpm:
            t_aero = self.aero_torque_model(max(rpm_val, 0), stage_index, bemt_results)
            if peak_torque > 0:
                sin_val = min(abs(t_aero) / peak_torque, 1.0)
                angle = math.degrees(math.asin(sin_val)) / n_poles
            else:
                angle = 0
            coupling_angles.append(angle)

        coupling_angles = np.array(coupling_angles)
        max_angle = float(np.max(coupling_angles)) if len(coupling_angles) > 0 else 0

        return TransientResult(
            time=sol.t,
            speed=speed_rpm,
            coupling_angle=coupling_angles,
            max_coupling_angle=max_angle,
            slip_angle=s_angle,
            coupling_margin_pct=(1 - max_angle / s_angle) * 100 if s_angle > 0 else 100,
        )

    def simulate_full_system(self, bemt_results=None) -> SystemSimResult:
        """Run full system simulation: shared motor equilibrium + transient."""
        operating_points = self.find_system_equilibrium(bemt_results)

        # Transient for first stage (most demanding)
        transient = self.simulate_startup(0, bemt_results)

        # Power budget summary
        total_power_in = sum(op.power_in for op in operating_points)
        total_power_thrust = sum(op.power_thrust for op in operating_points)
        total_losses = total_power_in - total_power_thrust

        power_budget = {
            "total_power_in_W": total_power_in,
            "total_power_thrust_W": total_power_thrust,
            "total_losses_W": total_losses,
            "overall_efficiency": total_power_thrust / total_power_in if total_power_in > 0 else 0,
        }

        # Convergence: motor RPM within 10% of design motor RPM
        design_motor_rpm = self.motor["rpm"]
        # All stages share the same motor — check via stage 0 (direct drive, ratio=1)
        eq_motor_rpm = operating_points[0].rpm * self.gear_ratio_for_stage(0)
        converged = abs(eq_motor_rpm - design_motor_rpm) / design_motor_rpm < 0.10

        return SystemSimResult(
            operating_points=operating_points,
            transient=transient,
            power_budget=power_budget,
            converged=converged,
        )
