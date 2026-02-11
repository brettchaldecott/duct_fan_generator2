"""Full power chain system simulation.

Simulates the complete power chain from motor through gears and magnetic
coupling to fan load. Finds steady-state operating point and simulates
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

        T = T_stall × (1 - RPM / RPM_no_load)
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

    def gear_efficiency(self) -> float:
        """Combined gear train efficiency."""
        eta_per_stage = self.gears["efficiency_per_stage"]
        n_stages = self.gears["num_stages"]
        return eta_per_stage ** n_stages

    def aero_torque_model(self, rpm: float, stage_index: int,
                          bemt_results=None) -> float:
        """Aerodynamic torque model.

        Uses quadratic scaling from BEMT reference point:
        Q(ω) = Q_ref × (ω / ω_ref)²
        """
        if bemt_results is not None and stage_index < len(bemt_results.stages):
            ref = bemt_results.stages[stage_index]
            ref_rpm = ref.rpm
            ref_torque = ref.total_torque
            if ref_rpm > 0:
                return ref_torque * (rpm / ref_rpm) ** 2
        # Fallback: rough estimate
        omega = rpm * 2 * math.pi / 60
        return 1e-8 * omega ** 2  # rough Q ∝ ω²

    def find_operating_point(self, stage_index: int,
                             bemt_results=None) -> OperatingPoint:
        """Find steady-state operating point using Brent's method.

        Finds RPM where motor torque (reflected through gears) equals
        aerodynamic load torque.
        """
        gear_ratio = self.derived["gear_ratio"]
        eta_gears = self.gear_efficiency()
        n_gear_stages = self.gears["num_stages"]

        # Total gear ratio to this stage
        total_ratio = gear_ratio ** min(stage_index, n_gear_stages)

        target_rpm = self.derived["stage_rpms"][stage_index]

        def torque_balance(fan_rpm):
            """Residual: motor torque reflected to fan - aero torque."""
            motor_rpm = fan_rpm * total_ratio
            if motor_rpm >= self.motor["no_load_rpm"]:
                return -1  # Motor can't reach this speed
            t_motor = self.motor_torque(motor_rpm)
            t_reflected = t_motor * total_ratio * eta_gears
            t_aero = self.aero_torque_model(fan_rpm, stage_index, bemt_results)
            return t_reflected - t_aero

        # Search for equilibrium
        rpm_min = 1.0
        rpm_max = target_rpm * 2

        try:
            # Check if solution exists in range
            f_min = torque_balance(rpm_min)
            f_max = torque_balance(rpm_max)

            if f_min * f_max > 0:
                # No sign change — use target RPM as approximation
                op_rpm = target_rpm
            else:
                op_rpm = brentq(torque_balance, rpm_min, rpm_max, xtol=1.0)
        except (ValueError, RuntimeError):
            op_rpm = target_rpm

        # Compute operating point details
        motor_rpm = op_rpm * total_ratio
        t_motor = self.motor_torque(motor_rpm)
        t_aero = self.aero_torque_model(op_rpm, stage_index, bemt_results)

        # Coupling analysis
        if stage_index < len(self.config["magnetic_coupling"]["stages"]):
            stage_cfg = self.config["magnetic_coupling"]["stages"][stage_index]
            peak_torque = self.coupling.estimate_peak_torque(stage_cfg)
            slip_angle = self.coupling.slip_angle(stage_cfg["num_pole_pairs"])

            if peak_torque > 0:
                # Coupling angle: T = T_peak × sin(n × θ)
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
            slip_angle = 90

        # Power budget
        omega_motor = motor_rpm * 2 * math.pi / 60
        omega_fan = op_rpm * 2 * math.pi / 60

        p_shaft = t_motor * omega_motor
        motor_eff = 0.85  # typical DC motor efficiency
        p_electrical = p_shaft / motor_eff if motor_eff > 0 else p_shaft
        p_after_gears = p_shaft * eta_gears
        p_thrust = t_aero * omega_fan

        efficiency = p_thrust / p_electrical if p_electrical > 0 else 0

        return OperatingPoint(
            stage_index=stage_index,
            rpm=op_rpm,
            motor_torque=t_motor,
            aero_torque=t_aero,
            coupling_torque_angle=coupling_angle,
            coupling_slip_pct=coupling_slip_pct,
            power_in=p_electrical,
            power_shaft=p_shaft,
            power_after_gears=p_after_gears,
            power_thrust=p_thrust,
            efficiency=efficiency,
        )

    def simulate_startup(self, stage_index: int = 0,
                         bemt_results=None,
                         t_final: float = 2.0) -> TransientResult:
        """Simulate startup transient to check coupling doesn't slip.

        Simple 1-DOF model: J × dω/dt = T_motor - T_load
        """
        gear_ratio = self.derived["gear_ratio"]
        eta_gears = self.gear_efficiency()

        # Moment of inertia estimate (rough)
        # J ≈ 0.5 × m × r² for blade ring
        blade_mat = self.config["materials"]["blade"]
        r_tip = self.derived["blade_tip_radius"] / 1000
        r_hub = self.derived["blade_hub_radius"] / 1000
        # Rough blade ring mass estimate: ~50g
        m_ring = 0.05  # kg
        J = 0.5 * m_ring * (r_tip**2 + r_hub**2)

        # Coupling parameters
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
            omega = state[0]  # rad/s
            rpm = omega * 60 / (2 * math.pi)

            # Motor torque reflected through gears
            motor_rpm = rpm * gear_ratio ** min(stage_index, self.gears["num_stages"])
            motor_rpm = max(0, min(motor_rpm, self.motor["no_load_rpm"] - 1))
            t_motor = self.motor_torque(motor_rpm)
            t_drive = t_motor * gear_ratio * eta_gears

            # Limit drive torque to coupling capacity
            t_drive = min(t_drive, peak_torque)

            # Aero load
            t_load = self.aero_torque_model(max(rpm, 0), stage_index, bemt_results)

            d_omega = (t_drive - t_load) / J
            return [d_omega]

        # Integrate
        sol = solve_ivp(
            dynamics,
            [0, t_final],
            [0.0],  # start from rest
            max_step=0.01,
            method='RK45',
        )

        speed_rpm = sol.y[0] * 60 / (2 * math.pi)

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
        """Run full system simulation: all stages + transient."""
        operating_points = []

        for i in range(len(self.config["blades"]["stages"])):
            op = self.find_operating_point(i, bemt_results)
            operating_points.append(op)

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

        converged = all(
            abs(op.rpm - self.derived["stage_rpms"][op.stage_index])
            / self.derived["stage_rpms"][op.stage_index] < 0.5  # within 50%
            for op in operating_points
        )

        return SystemSimResult(
            operating_points=operating_points,
            transient=transient,
            power_budget=power_budget,
            converged=converged,
        )
