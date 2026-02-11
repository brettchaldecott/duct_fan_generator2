"""Shared geometry helpers for CAD generation."""

import math
import numpy as np
from typing import List, Tuple


def circle_points(radius: float, n_points: int = 64,
                  center: Tuple[float, float] = (0, 0)) -> List[Tuple[float, float]]:
    """Generate points on a circle.

    Args:
        radius: Circle radius in mm
        n_points: Number of points
        center: Center coordinates (x, y)

    Returns:
        List of (x, y) tuples
    """
    angles = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    return [
        (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
        for a in angles
    ]


def bolt_circle_points(pcd: float, n_holes: int,
                       start_angle: float = 0) -> List[Tuple[float, float]]:
    """Generate bolt hole positions on a pitch circle diameter.

    Args:
        pcd: Pitch circle diameter in mm
        n_holes: Number of holes
        start_angle: Starting angle in degrees

    Returns:
        List of (x, y) center positions
    """
    r = pcd / 2
    start_rad = math.radians(start_angle)
    return [
        (r * math.cos(start_rad + 2 * math.pi * i / n_holes),
         r * math.sin(start_rad + 2 * math.pi * i / n_holes))
        for i in range(n_holes)
    ]


def annular_area(r_outer: float, r_inner: float) -> float:
    """Compute annular area."""
    return math.pi * (r_outer**2 - r_inner**2)


def polar_pattern(points: np.ndarray, n_copies: int,
                  axis_center: Tuple[float, float] = (0, 0)) -> List[np.ndarray]:
    """Create polar pattern of 2D points.

    Args:
        points: (N, 2) array of x, y coordinates
        n_copies: Total number of copies (including original)
        axis_center: Center of rotation

    Returns:
        List of (N, 2) arrays, one per copy
    """
    results = []
    for i in range(n_copies):
        angle = 2 * math.pi * i / n_copies
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        cx, cy = axis_center

        rotated = np.column_stack([
            (points[:, 0] - cx) * cos_a - (points[:, 1] - cy) * sin_a + cx,
            (points[:, 0] - cx) * sin_a + (points[:, 1] - cy) * cos_a + cy,
        ])
        results.append(rotated)

    return results
