"""Tests for full power chain system simulation."""

import pytest
from validation.system_simulator import SystemSimulator


class TestSteadyState:
    """Test steady-state operating point determination."""

    def test_operating_points_computed(self, default_config, bemt_results):
        """Operating points are computed for all stages."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_full_system(bemt_results)
        assert len(result.operating_points) == len(default_config["blades"]["stages"])

    def test_operating_rpms_positive(self, default_config, bemt_results):
        """All operating RPMs are positive and near design targets."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_full_system(bemt_results)
        for op in result.operating_points:
            assert op.rpm > 0, f"Stage {op.stage_index}: RPM={op.rpm}"
            target = default_config["derived"]["stage_rpms"][op.stage_index]
            error_pct = abs(op.rpm - target) / target
            assert error_pct < 0.20, (
                f"Stage {op.stage_index}: RPM={op.rpm:.0f} is {error_pct*100:.0f}% "
                f"from target {target:.0f}"
            )

    def test_power_in_positive(self, default_config, bemt_results):
        """All stages have positive power input."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_full_system(bemt_results)
        for op in result.operating_points:
            assert op.power_in > 0

    def test_coupling_angle_within_range(self, default_config, bemt_results):
        """Coupling angle should be between 0 and slip angle."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_full_system(bemt_results)
        for op in result.operating_points:
            assert op.coupling_torque_angle >= 0
            assert op.coupling_slip_pct < 100, (
                f"Stage {op.stage_index}: coupling at {op.coupling_slip_pct:.1f}% of slip"
            )

    def test_convergence_status_reported(self, default_config, bemt_results):
        """System converges with properly sized motor."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_full_system(bemt_results)
        assert result.converged is True, (
            f"System did not converge. Operating RPMs: "
            f"{[op.rpm for op in result.operating_points]}, "
            f"targets: {default_config['derived']['stage_rpms']}"
        )

    def test_shared_motor_equilibrium(self, default_config, bemt_results):
        """All stages derive RPM from the same motor RPM via gear ratios."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_full_system(bemt_results)
        ops = result.operating_points
        # Recover motor RPM from each stage
        motor_rpms = []
        for op in ops:
            ratio = sim.gear_ratio_for_stage(op.stage_index)
            motor_rpms.append(op.rpm * ratio)
        # All should agree (same motor)
        for m_rpm in motor_rpms:
            assert abs(m_rpm - motor_rpms[0]) < 1.0, (
                f"Motor RPMs disagree: {motor_rpms}"
            )


class TestPerStageEfficiency:
    """Test per-stage gear efficiency."""

    def test_stage_0_no_gear_loss(self, default_config):
        """Stage 0 (direct drive) has efficiency 1.0."""
        sim = SystemSimulator(default_config)
        assert sim.gear_efficiency_for_stage(0) == 1.0

    def test_stage_1_one_gear_stage(self, default_config):
        """Stage 1 traverses 1 gear stage -> eta = 0.95."""
        sim = SystemSimulator(default_config)
        eta = default_config["gears"]["efficiency_per_stage"]
        assert sim.gear_efficiency_for_stage(1) == pytest.approx(eta)

    def test_stage_2_two_gear_stages(self, default_config):
        """Stage 2 traverses 2 gear stages -> eta = 0.95^2."""
        sim = SystemSimulator(default_config)
        eta = default_config["gears"]["efficiency_per_stage"]
        assert sim.gear_efficiency_for_stage(2) == pytest.approx(eta ** 2)

    def test_gear_ratio_stage_0(self, default_config):
        """Stage 0 has gear ratio 1.0 (direct drive)."""
        sim = SystemSimulator(default_config)
        assert sim.gear_ratio_for_stage(0) == 1.0

    def test_gear_ratio_stage_1(self, default_config):
        """Stage 1 has gear ratio = gear_ratio^1."""
        sim = SystemSimulator(default_config)
        assert sim.gear_ratio_for_stage(1) == pytest.approx(
            default_config["derived"]["gear_ratio"]
        )

    def test_gear_ratio_stage_2(self, default_config):
        """Stage 2 has gear ratio = gear_ratio^2."""
        sim = SystemSimulator(default_config)
        gr = default_config["derived"]["gear_ratio"]
        assert sim.gear_ratio_for_stage(2) == pytest.approx(gr ** 2)


class TestMotorModel:
    """Test motor torque-speed model."""

    def test_stall_torque(self, default_config):
        """At 0 RPM, torque equals stall torque."""
        sim = SystemSimulator(default_config)
        t_stall = default_config["motor"]["stall_torque"]
        assert abs(sim.motor_torque(0) - t_stall) < 1e-10

    def test_no_load_zero_torque(self, default_config):
        """At no-load RPM, torque equals zero."""
        sim = SystemSimulator(default_config)
        rpm_nl = default_config["motor"]["no_load_rpm"]
        assert abs(sim.motor_torque(rpm_nl)) < 1e-10

    def test_torque_decreases_with_rpm(self, default_config):
        """Torque decreases linearly with RPM."""
        sim = SystemSimulator(default_config)
        t1 = sim.motor_torque(3000)
        t2 = sim.motor_torque(6000)
        t3 = sim.motor_torque(9000)
        assert t1 > t2 > t3 > 0


class TestTransient:
    """Test startup transient simulation."""

    def test_transient_runs(self, default_config, bemt_results):
        """Transient simulation runs without error."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_startup(0, bemt_results, t_final=1.0)
        assert len(result.time) > 0
        assert len(result.speed) == len(result.time)

    def test_starts_from_rest(self, default_config, bemt_results):
        """System starts from 0 RPM."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_startup(0, bemt_results, t_final=1.0)
        assert abs(result.speed[0]) < 1e-6

    def test_speed_increases(self, default_config, bemt_results):
        """Speed should increase during startup."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_startup(0, bemt_results, t_final=1.0)
        assert result.speed[-1] > result.speed[0]

    def test_coupling_margin_positive(self, default_config, bemt_results):
        """Coupling margin stays positive during startup."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_startup(0, bemt_results, t_final=1.0)
        assert result.coupling_margin_pct > 0, (
            f"Coupling margin = {result.coupling_margin_pct:.1f}% "
            f"(max angle: {result.max_coupling_angle:.1f}deg, "
            f"slip angle: {result.slip_angle:.1f}deg)"
        )


class TestPowerBudget:
    """Test power budget accounting."""

    def test_efficiency_bounded(self, default_config, bemt_results):
        """Overall efficiency is between 0 and 1."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_full_system(bemt_results)
        eta = result.power_budget["overall_efficiency"]
        assert 0 <= eta <= 1.0, f"Efficiency {eta} out of bounds"

    def test_power_balance(self, default_config, bemt_results):
        """Power in >= power thrust + losses."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_full_system(bemt_results)
        pb = result.power_budget
        assert pb["total_power_in_W"] >= pb["total_power_thrust_W"] - 0.01

    def test_losses_non_negative(self, default_config, bemt_results):
        """System losses should be non-negative."""
        sim = SystemSimulator(default_config)
        result = sim.simulate_full_system(bemt_results)
        assert result.power_budget["total_losses_W"] >= -0.01
