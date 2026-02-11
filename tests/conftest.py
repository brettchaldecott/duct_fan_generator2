"""Shared test fixtures, tolerance helpers, and cached results."""

import os
import pytest
from src.config import load_config, DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------

def assert_stress_safe(actual_stress: float, material_limit: float, safety_factor: float = 3.0):
    """Assert that actual stress is within the material limit with a safety factor.

    Args:
        actual_stress: Computed stress in MPa
        material_limit: Material strength in MPa
        safety_factor: Required safety factor (default 3.0)
    """
    allowable = material_limit / safety_factor
    assert actual_stress <= allowable, (
        f"Stress {actual_stress:.2f} MPa exceeds allowable "
        f"{allowable:.2f} MPa (limit={material_limit} MPa, SF={safety_factor})"
    )


def assert_within_range(value: float, min_val: float, max_val: float, unit: str = "mm"):
    """Assert that a value falls within [min_val, max_val].

    Args:
        value: Measured value
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        unit: Unit string for error message
    """
    assert min_val <= value <= max_val, (
        f"Value {value:.4f} {unit} is outside range "
        f"[{min_val:.4f}, {max_val:.4f}] {unit}"
    )


def assert_clearance(gap: float, minimum: float, description: str = ""):
    """Assert that a clearance gap meets the minimum requirement.

    Args:
        gap: Measured clearance in mm
        minimum: Minimum required clearance in mm
        description: Optional description for error messages
    """
    desc = f" ({description})" if description else ""
    assert gap >= minimum, (
        f"Clearance{desc}: {gap:.3f} mm is less than minimum {minimum:.3f} mm"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_config():
    """Load the default configuration (session-scoped, loaded once)."""
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="session")
def bemt_results(default_config):
    """Pre-computed BEMT analysis for default config (session-scoped)."""
    from src.bemt import BEMTSolver
    solver = BEMTSolver(default_config)
    return solver.solve_all_stages()


@pytest.fixture(scope="session")
def generated_meshes(default_config, bemt_results):
    """Dict of part_name -> trimesh.Trimesh for all generated parts.

    Session-scoped so CAD generation only happens once per test session.
    Returns empty dict if assembly generator is not available or fails.
    """
    try:
        from src.assembly import AssemblyGenerator
        generator = AssemblyGenerator(default_config, bemt_results)
        return generator.generate_all_meshes()
    except (ImportError, Exception):
        return {}
