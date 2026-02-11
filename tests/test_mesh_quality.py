"""Tests for mesh quality validation.

Tests mesh validation logic using simple test geometries.
Full STL validation of generated parts requires CadQuery geometry generation
and is tested via the generated_meshes fixture.
"""

import pytest
import numpy as np
import trimesh
from validation.mesh_validator import MeshValidator


@pytest.fixture
def mesh_validator(default_config):
    """Create mesh validator with default config."""
    return MeshValidator(default_config)


@pytest.fixture
def good_mesh():
    """A valid watertight box mesh."""
    return trimesh.creation.box(extents=[50, 50, 50])


@pytest.fixture
def bad_mesh():
    """A non-watertight mesh (open surface)."""
    # Create a simple triangle that's not a closed solid
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])
    faces = np.array([[0, 1, 2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


class TestMeshValidation:
    """Test mesh quality validation."""

    def test_valid_mesh_passes(self, mesh_validator, good_mesh):
        """A watertight box should pass all checks."""
        result = mesh_validator.validate_mesh(good_mesh, "test_box")
        assert result.is_watertight
        assert result.is_volume
        assert result.fits_build_volume
        assert result.passed

    def test_invalid_mesh_fails(self, mesh_validator, bad_mesh):
        """A non-watertight mesh should fail."""
        result = mesh_validator.validate_mesh(bad_mesh, "test_open")
        assert not result.is_watertight
        assert not result.passed

    def test_oversized_mesh_fails(self, mesh_validator):
        """A mesh exceeding build volume should fail."""
        huge_box = trimesh.creation.box(extents=[400, 400, 400])
        result = mesh_validator.validate_mesh(huge_box, "huge_box")
        assert not result.fits_build_volume
        assert not result.passed

    def test_volume_computed(self, mesh_validator, good_mesh):
        """Volume should be computed for valid meshes."""
        result = mesh_validator.validate_mesh(good_mesh, "box")
        assert result.volume_mm3 > 0
        # 50x50x50 box = 125000 mm³
        assert abs(result.volume_mm3 - 125000) < 1

    def test_bounding_box_computed(self, mesh_validator, good_mesh):
        """Bounding box dimensions should be computed."""
        result = mesh_validator.validate_mesh(good_mesh, "box")
        assert len(result.bounding_box) == 3
        for dim in result.bounding_box:
            assert abs(dim - 50) < 0.1  # 50mm box

    def test_details_always_present(self, mesh_validator, good_mesh):
        """Result always has details list."""
        result = mesh_validator.validate_mesh(good_mesh, "box")
        assert len(result.details) > 0


class TestMultipleMeshValidation:
    """Test batch mesh validation."""

    def test_validate_all_meshes(self, mesh_validator):
        """validate_all_meshes processes multiple meshes."""
        meshes = {
            "box_a": trimesh.creation.box(extents=[50, 50, 50]),
            "box_b": trimesh.creation.box(extents=[100, 100, 100]),
        }
        results = mesh_validator.validate_all_meshes(meshes)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_mixed_quality_meshes(self, mesh_validator, good_mesh, bad_mesh):
        """Mixed quality meshes: good passes, bad fails."""
        meshes = {
            "good": good_mesh,
            "bad": bad_mesh,
        }
        results = mesh_validator.validate_all_meshes(meshes)
        good_result = next(r for r in results if r.part_name == "good")
        bad_result = next(r for r in results if r.part_name == "bad")
        assert good_result.passed
        assert not bad_result.passed


class TestGeneratedMeshes:
    """Test mesh quality of actually generated parts.

    These tests require the AssemblyGenerator and CadQuery.
    They are skipped if no meshes are available.
    """

    def test_generated_meshes_exist(self, generated_meshes):
        """If assembly generator works, at least some meshes are generated."""
        if not generated_meshes:
            pytest.skip("No generated meshes available (AssemblyGenerator not ready)")
        assert len(generated_meshes) > 0

    def test_generated_meshes_fit_build(self, generated_meshes, mesh_validator):
        """If meshes are generated, they should fit build volume."""
        if not generated_meshes:
            pytest.skip("No generated meshes available (AssemblyGenerator not ready)")
        for name, mesh in generated_meshes.items():
            result = mesh_validator.validate_mesh(mesh, name)
            assert result.fits_build_volume, f"{name} exceeds build volume"

    def test_generated_meshes_quality(self, generated_meshes, mesh_validator):
        """Report mesh quality for all generated parts (informational)."""
        if not generated_meshes:
            pytest.skip("No generated meshes available (AssemblyGenerator not ready)")
        watertight_count = 0
        for name, mesh in generated_meshes.items():
            result = mesh_validator.validate_mesh(mesh, name)
            if result.is_watertight:
                watertight_count += 1
        # At least most parts should be watertight
        assert watertight_count >= len(generated_meshes) * 0.7, (
            f"Only {watertight_count}/{len(generated_meshes)} meshes are watertight"
        )
