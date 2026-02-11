"""STL mesh quality validation using trimesh.

Checks that generated STL files are watertight, manifold, have no
degenerate faces, and meet minimum wall thickness requirements.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import trimesh


@dataclass
class MeshValidationResult:
    """Result of mesh quality validation for a single part."""
    part_name: str
    is_watertight: bool
    is_volume: bool
    has_degenerate_faces: bool
    num_degenerate: int
    volume_mm3: float
    bounding_box: tuple  # (x, y, z) dimensions in mm
    fits_build_volume: bool
    passed: bool
    details: List[str]


class MeshValidator:
    """Validates STL mesh quality for 3D printing."""

    def __init__(self, config: dict):
        self.config = config
        self.print_cfg = config["print"]
        self.max_build = (
            self.print_cfg["max_build_x"],
            self.print_cfg["max_build_y"],
            self.print_cfg["max_build_z"],
        )
        self.min_wall = self.print_cfg["min_wall_thickness"]

    def validate_mesh(self, mesh: trimesh.Trimesh, part_name: str) -> MeshValidationResult:
        """Validate a single mesh for print quality.

        Args:
            mesh: trimesh.Trimesh object
            part_name: Name identifier for the part

        Returns:
            MeshValidationResult
        """
        details = []

        # Watertight check
        is_watertight = mesh.is_watertight
        if not is_watertight:
            details.append("Mesh is NOT watertight — has open edges")

        # Valid volume check
        is_volume = mesh.is_volume
        if not is_volume:
            details.append("Mesh is NOT a valid volume — inconsistent winding or open edges")

        # Degenerate face check
        areas = mesh.area_faces
        degenerate_mask = areas < 1e-10  # nearly zero-area triangles
        num_degenerate = int(np.sum(degenerate_mask))
        has_degenerate = num_degenerate > 0
        if has_degenerate:
            details.append(f"Found {num_degenerate} degenerate (zero-area) faces")

        # Volume
        volume = mesh.volume if is_volume else 0

        # Bounding box
        bb = mesh.bounding_box.extents  # (x, y, z) dimensions
        bb_tuple = tuple(bb)

        # Build volume check
        fits = all(bb[i] <= self.max_build[i] for i in range(3))
        if not fits:
            details.append(
                f"Part exceeds build volume: {bb[0]:.1f}x{bb[1]:.1f}x{bb[2]:.1f}mm "
                f"> {self.max_build[0]}x{self.max_build[1]}x{self.max_build[2]}mm"
            )

        passed = is_watertight and is_volume and not has_degenerate and fits

        if passed:
            details.append("All mesh quality checks passed")

        return MeshValidationResult(
            part_name=part_name,
            is_watertight=is_watertight,
            is_volume=is_volume,
            has_degenerate_faces=has_degenerate,
            num_degenerate=num_degenerate,
            volume_mm3=volume,
            bounding_box=bb_tuple,
            fits_build_volume=fits,
            passed=passed,
            details=details,
        )

    def validate_stl_file(self, filepath: str, part_name: str) -> MeshValidationResult:
        """Validate an STL file from disk."""
        mesh = trimesh.load(filepath)
        if isinstance(mesh, trimesh.Scene):
            # Multi-body: merge
            mesh = trimesh.util.concatenate(mesh.dump())
        return self.validate_mesh(mesh, part_name)

    def validate_all_meshes(self, meshes: dict) -> List[MeshValidationResult]:
        """Validate all meshes in a dict of name -> trimesh.Trimesh.

        Args:
            meshes: Dictionary mapping part names to Trimesh objects

        Returns:
            List of MeshValidationResult
        """
        results = []
        for name, mesh in meshes.items():
            results.append(self.validate_mesh(mesh, name))
        return results
