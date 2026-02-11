"""Tests for magnetic coupling analysis."""

import math
import pytest
import numpy as np
from src.magnetic_coupling import MagneticCoupling
from validation.coupling_validator import CouplingValidator


class TestCouplingAnalysis:
    """Test magnetic coupling torque estimation."""

    def test_peak_torque_positive(self, default_config):
        """Peak torque should be positive for all stages."""
        coupling = MagneticCoupling(default_config)
        for stage_cfg in default_config["magnetic_coupling"]["stages"]:
            T = coupling.estimate_peak_torque(stage_cfg)
            assert T > 0, f"Non-positive peak torque: {T}"

    def test_thicker_magnets_more_torque(self, default_config):
        """Thicker magnets should produce more torque."""
        coupling = MagneticCoupling(default_config)
        stages = default_config["magnetic_coupling"]["stages"]
        # Stage 0: 3mm thick, Stage 1: 5mm thick (same diameter)
        t0 = coupling.estimate_peak_torque(stages[0])
        t1 = coupling.estimate_peak_torque(stages[1])
        assert t1 > t0, "5mm magnets should produce more torque than 3mm"

    def test_slip_angle_correct(self, default_config):
        """Slip angle = 90° / num_pole_pairs."""
        coupling = MagneticCoupling(default_config)
        for stage in default_config["magnetic_coupling"]["stages"]:
            n = stage["num_pole_pairs"]
            expected = 90.0 / n
            actual = coupling.slip_angle(n)
            assert abs(actual - expected) < 1e-10


class TestTorqueCurve:
    """Test torque-angle curve generation."""

    def test_curve_shape(self, default_config):
        """Torque curve should be sinusoidal, peaking at slip angle."""
        coupling = MagneticCoupling(default_config)
        stage_cfg = default_config["magnetic_coupling"]["stages"][0]
        curve = coupling.torque_curve(stage_cfg)

        assert curve.shape[1] == 2  # angle, torque columns
        assert curve.shape[0] > 10  # enough points

        # Peak should be near the end (at slip angle = end of half-cycle)
        peak_idx = np.argmax(curve[:, 1])
        # Peak should be in the upper half of the curve
        # For sin(0..π/2), peak is at the last point
        assert peak_idx >= curve.shape[0] * 0.4

    def test_curve_starts_at_zero(self, default_config):
        """Torque should be zero at zero angle."""
        coupling = MagneticCoupling(default_config)
        stage_cfg = default_config["magnetic_coupling"]["stages"][0]
        curve = coupling.torque_curve(stage_cfg)
        assert abs(curve[0, 1]) < 1e-6  # first point should be ~0

    def test_peak_matches_estimate(self, default_config):
        """Peak of torque curve matches estimated peak torque."""
        coupling = MagneticCoupling(default_config)
        stage_cfg = default_config["magnetic_coupling"]["stages"][0]
        curve = coupling.torque_curve(stage_cfg, num_points=200)
        peak_from_curve = np.max(curve[:, 1])
        peak_estimated = coupling.estimate_peak_torque(stage_cfg)
        assert abs(peak_from_curve - peak_estimated) / peak_estimated < 0.02


class TestCouplingValidator:
    """Test coupling validation framework."""

    def test_validator_runs(self, default_config):
        """Coupling validator runs with BEMT results."""
        validator = CouplingValidator(default_config)
        # Use small test torques
        result = validator.validate([0.01, 0.01, 0.01])
        assert result is not None
        assert len(result.stages) == 3

    def test_validator_with_bemt(self, default_config, bemt_results):
        """Coupling validator runs with actual BEMT results."""
        validator = CouplingValidator(default_config)
        result = validator.validate_with_bemt(bemt_results)
        assert len(result.stages) == len(bemt_results.stages)

    def test_critical_stage_identified(self, default_config, bemt_results):
        """Critical stage (lowest margin) is identified."""
        validator = CouplingValidator(default_config)
        result = validator.validate_with_bemt(bemt_results)
        # Critical stage should have the lowest safety factor
        sfs = [s.safety_factor for s in result.stages]
        assert result.critical_stage == sfs.index(min(sfs))

    def test_ferromagnetic_warning(self, default_config):
        """Ferromagnetic warning is generated."""
        validator = CouplingValidator(default_config)
        result = validator.validate([0.01, 0.01, 0.01])
        assert "PETG CF" in result.ferromagnetic_warning

    def test_torque_margins_reported(self, default_config, bemt_results):
        """Each stage reports torque margin details."""
        validator = CouplingValidator(default_config)
        result = validator.validate_with_bemt(bemt_results)
        for stage in result.stages:
            assert stage.peak_torque > 0
            assert stage.required_torque > 0
            assert stage.safety_factor > 0


class TestMagnetPockets:
    """Test magnet pocket geometry generation."""

    def test_pocket_specs(self, default_config):
        """Magnet pocket specs are generated for each stage."""
        coupling = MagneticCoupling(default_config)
        for i in range(len(default_config["magnetic_coupling"]["stages"])):
            spec = coupling.magnet_pocket_specs(i)
            assert spec["pocket_diameter"] > spec["magnet_diameter"]
            assert spec["pocket_depth"] > spec["magnet_thickness"]
            assert spec["num_pockets"] > 0

    def test_pocket_clearance(self, default_config):
        """Pocket diameter includes configured clearance."""
        coupling = MagneticCoupling(default_config)
        clearance = default_config["magnetic_coupling"]["magnet_pocket_clearance"]
        for i, stage in enumerate(default_config["magnetic_coupling"]["stages"]):
            spec = coupling.magnet_pocket_specs(i)
            expected_d = stage["magnet_diameter"] + 2 * clearance
            assert abs(spec["pocket_diameter"] - expected_d) < 1e-10
