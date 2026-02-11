"""NACA 4-digit airfoil coordinate generation.

Implements the standard NACA 4-digit airfoil equations for mean camber line
and thickness distribution. Supports blended profiles for root-to-tip
interpolation on fan blades.
"""

import numpy as np


def parse_naca4(designation: str) -> tuple:
    """Parse a NACA 4-digit designation string.

    Args:
        designation: e.g. "NACA4412" or "NACA0012"

    Returns:
        (max_camber, camber_position, max_thickness) as fractions of chord
        e.g. NACA4412 -> (0.04, 0.4, 0.12)
    """
    s = designation.upper().replace("NACA", "").strip()
    if len(s) != 4:
        raise ValueError(f"Invalid NACA 4-digit designation: '{designation}'")
    m = int(s[0]) / 100.0    # max camber as fraction of chord
    p = int(s[1]) / 10.0     # position of max camber as fraction of chord
    t = int(s[2:]) / 100.0   # max thickness as fraction of chord
    return m, p, t


def naca4_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """Compute NACA 4-digit thickness distribution.

    Args:
        x: Chordwise positions (0 to 1), normalized by chord
        t: Maximum thickness as fraction of chord

    Returns:
        Half-thickness at each x position
    """
    return (t / 0.20) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4  # closed trailing edge coefficient
    )


def naca4_camber(x: np.ndarray, m: float, p: float) -> tuple:
    """Compute NACA 4-digit mean camber line and its gradient.

    Args:
        x: Chordwise positions (0 to 1)
        m: Maximum camber as fraction of chord
        p: Position of maximum camber as fraction of chord

    Returns:
        (yc, dyc_dx): camber line y-coordinates and gradient
    """
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    if m == 0 or p == 0:
        # Symmetric airfoil
        return yc, dyc

    # Forward of max camber
    front = x <= p
    yc[front] = (m / p**2) * (2 * p * x[front] - x[front]**2)
    dyc[front] = (2 * m / p**2) * (p - x[front])

    # Aft of max camber
    rear = x > p
    yc[rear] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[rear] - x[rear]**2)
    dyc[rear] = (2 * m / (1 - p)**2) * (p - x[rear])

    return yc, dyc


def generate_naca4_profile(designation: str, num_points: int = 100,
                           chord_length: float = 1.0) -> np.ndarray:
    """Generate a closed NACA 4-digit airfoil profile.

    Points are distributed using cosine spacing for better leading-edge
    resolution. The profile is closed (first point == last point).

    Args:
        designation: NACA designation (e.g. "NACA4412")
        num_points: Number of points per surface (total = 2*num_points - 1 + closure)
        chord_length: Physical chord length in mm

    Returns:
        numpy array of shape (N, 2) with (x, y) coordinates, closed profile
    """
    m, p, t = parse_naca4(designation)

    # Cosine-spaced x distribution for better LE resolution
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))

    # Thickness distribution
    yt = naca4_thickness(x, t)

    # Camber line
    yc, dyc = naca4_camber(x, m, p)

    # Angle of camber line
    theta = np.arctan(dyc)

    # Upper surface
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    # Lower surface
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Build closed profile: upper surface (LE to TE) + lower surface reversed (TE to LE)
    # Skip the duplicate LE point on the lower surface
    x_profile = np.concatenate([xu, xl[-2::-1]])
    y_profile = np.concatenate([yu, yl[-2::-1]])

    # Close the profile (first point = last point)
    if not (np.isclose(x_profile[0], x_profile[-1]) and
            np.isclose(y_profile[0], y_profile[-1])):
        x_profile = np.append(x_profile, x_profile[0])
        y_profile = np.append(y_profile, y_profile[0])

    # Scale to chord length
    coords = np.column_stack([x_profile * chord_length, y_profile * chord_length])
    return coords


def blend_profiles(profile_root: np.ndarray, profile_tip: np.ndarray,
                   fraction: float) -> np.ndarray:
    """Linearly interpolate between two airfoil profiles.

    Both profiles must have the same number of points.

    Args:
        profile_root: Root airfoil coordinates (N, 2)
        profile_tip: Tip airfoil coordinates (N, 2)
        fraction: 0.0 = root, 1.0 = tip

    Returns:
        Blended profile coordinates (N, 2)
    """
    if profile_root.shape != profile_tip.shape:
        raise ValueError(
            f"Profile shapes must match: {profile_root.shape} vs {profile_tip.shape}"
        )
    return profile_root * (1 - fraction) + profile_tip * fraction
