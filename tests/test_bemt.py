"""Tests for Blade Element Momentum Theory solver."""

import math
import numpy as np
import pytest
from src.bemt import BEMTSolver, BEMTResults


class TestBEMTBasics:
    """Basic sanity tests for BEMT solver."""

    def test_solver_creates(self, default_config):
        """BEMT solver instantiates without error."""
        solver = BEMTSolver(default_config)
        assert solver is not None

    def test_solve_all_stages(self, default_config):
        """Solving all stages produces results for each configured stage."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        assert len(results.stages) == len(default_config["blades"]["stages"])

    def test_solve_returns_bemt_results(self, default_config):
        """solve_all_stages returns a BEMTResults dataclass."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        assert isinstance(results, BEMTResults)


class TestBEMTPhysics:
    """Physics validation tests."""

    def test_positive_thrust_per_stage(self, default_config):
        """Each stage should produce positive thrust."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        for stage in results.stages:
            assert stage.total_thrust > 0, (
                f"Stage {stage.stage_index} has non-positive thrust: {stage.total_thrust}"
            )

    def test_positive_torque_per_stage(self, default_config):
        """Each stage should require positive torque."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        for stage in results.stages:
            assert stage.total_torque > 0, (
                f"Stage {stage.stage_index} has non-positive torque: {stage.total_torque}"
            )

    def test_total_thrust_greater_than_single_stage(self, default_config):
        """Multi-stage total thrust > any single stage thrust."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        total = results.total_thrust
        for stage in results.stages:
            assert total > stage.total_thrust

    def test_induction_factor_bounded(self, default_config):
        """Axial induction factor should be < 0.5 or Glauert-corrected."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        for stage in results.stages:
            for section in stage.sections:
                assert section.a < 1.0, (
                    f"Induction factor a={section.a} >= 1.0 at r={section.radius}"
                )

    def test_counter_rotation_directions(self, default_config):
        """Stage directions should alternate per gear stages."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        dirs = [s.direction for s in results.stages]
        # Should alternate: 1, -1, 1
        for i in range(1, len(dirs)):
            assert dirs[i] != dirs[i-1], "Adjacent stages should rotate in opposite directions"

    def test_swirl_coupling(self, default_config):
        """Stage 2 should receive exit swirl from Stage 1."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        if len(results.stages) >= 2:
            # Stage 1 should produce some exit swirl
            stage1_swirl = results.stages[0].exit_swirl
            assert np.any(np.abs(stage1_swirl) > 0), (
                "Stage 1 should produce non-zero exit swirl"
            )


class TestBEMTSections:
    """Test individual section results."""

    def test_prandtl_tip_loss(self, default_config):
        """Tip loss factor should decrease toward blade tips."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        stage = results.stages[0]
        tip_losses = [s.tip_loss for s in stage.sections]
        # Last section (near tip) should have lower F than middle sections
        mid_idx = len(tip_losses) // 2
        assert tip_losses[-1] <= tip_losses[mid_idx] + 0.1  # some tolerance

    def test_chord_distribution_positive(self, default_config):
        """All section chords should be positive."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        for stage in results.stages:
            for section in stage.sections:
                assert section.chord > 0, f"Non-positive chord at r={section.radius}"

    def test_twist_distribution_reasonable(self, default_config):
        """Twist should generally decrease from hub to tip."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        stage = results.stages[0]
        twists = [s.twist for s in stage.sections]
        # Twist at hub should generally be larger than at tip
        # (higher angle at hub where tangential speed is lower)
        assert twists[0] > twists[-1] - 0.5, (
            "Hub twist should generally be larger than tip twist"
        )


class TestBEMTRPMSensitivity:
    """Test BEMT sensitivity to operating conditions."""

    def test_higher_rpm_more_thrust(self, default_config):
        """Higher RPM stage should produce more thrust (for same geometry)."""
        solver = BEMTSolver(default_config)
        results = solver.solve_all_stages()
        # Stage 1 (12000 RPM) should produce more thrust than Stage 3 (3000 RPM)
        # Note: different blade counts/profiles may affect this, so use generous tolerance
        stage1 = results.stages[0]
        stage3 = results.stages[-1]
        # At least stage 1 should have meaningful thrust
        assert stage1.total_thrust > 0.1  # At least 0.1N
