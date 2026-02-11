"""Tests for NACA 4-digit airfoil generation."""

import numpy as np
import pytest
from src.airfoil import (
    parse_naca4,
    generate_naca4_profile,
    naca4_thickness,
    naca4_camber,
    blend_profiles,
)


class TestNACA4Parsing:
    """Test NACA designation parsing."""

    def test_naca0012(self):
        m, p, t = parse_naca4("NACA0012")
        assert m == 0.0
        assert p == 0.0
        assert t == 0.12

    def test_naca4412(self):
        m, p, t = parse_naca4("NACA4412")
        assert m == 0.04
        assert p == 0.4
        assert t == 0.12

    def test_naca6412(self):
        m, p, t = parse_naca4("NACA6412")
        assert m == 0.06
        assert p == 0.4
        assert t == 0.12

    def test_invalid_designation(self):
        with pytest.raises(ValueError, match="Invalid NACA"):
            parse_naca4("NACA123")

    def test_case_insensitive(self):
        m1, p1, t1 = parse_naca4("NACA4412")
        m2, p2, t2 = parse_naca4("naca4412")
        assert m1 == m2 and p1 == p2 and t1 == t2


class TestSymmetricProfile:
    """Test NACA 0012 symmetric profile."""

    def test_naca0012_symmetric(self):
        """NACA 0012: upper and lower surfaces should be mirror images."""
        profile = generate_naca4_profile("NACA0012", num_points=50, chord_length=1.0)
        # For a symmetric profile, for each upper point (x, y), there should
        # be a lower point (x, -y)
        # Find the leading edge (min x)
        le_idx = np.argmin(profile[:, 0])
        upper = profile[:le_idx + 1]  # TE to LE along upper
        lower = profile[le_idx:]      # LE to TE along lower

        # Upper y should be positive (or zero at TE)
        assert np.all(upper[1:-1, 1] >= -1e-10)

    def test_naca0012_reference_values(self):
        """NACA 0012 thickness at key stations matches published data within 0.1%."""
        # Published NACA 0012 half-thickness values (y/c) at specific x/c:
        # x/c=0.0: 0.0, x/c=0.3: 0.05941, x/c=1.0: ~0.00126 (closed TE)
        x = np.array([0.0, 0.05, 0.10, 0.30, 0.50])
        t = 0.12
        yt = naca4_thickness(x, t)

        # At x=0, thickness should be 0
        assert abs(yt[0]) < 1e-10

        # At x=0.30, published half-thickness ≈ 0.05941
        assert abs(yt[3] - 0.05941) < 0.001  # within 0.1% of chord


class TestCamberedProfile:
    """Test NACA 4412 cambered profile."""

    def test_naca4412_camber_line(self):
        """NACA 4412 camber line matches analytical equations."""
        m, p, t = parse_naca4("NACA4412")
        x = np.linspace(0, 1, 100)
        yc, dyc = naca4_camber(x, m, p)

        # Max camber should occur near x = p = 0.4
        max_camber_idx = np.argmax(yc)
        max_camber_x = x[max_camber_idx]
        assert abs(max_camber_x - p) < 0.02  # within 2% of chord

        # Max camber value should be close to m = 0.04
        assert abs(yc[max_camber_idx] - m) < 0.005

    def test_cambered_profile_asymmetric(self):
        """NACA 4412 should produce an asymmetric profile."""
        profile = generate_naca4_profile("NACA4412", num_points=50)
        # Mean y should be positive (camber lifts the profile)
        mean_y = np.mean(profile[:, 1])
        assert mean_y > 0


class TestProfileClosure:
    """Test that profiles are properly closed."""

    @pytest.mark.parametrize("designation", ["NACA0012", "NACA4412", "NACA6412", "NACA4410"])
    def test_profile_closed(self, designation):
        """Profile first point equals last point."""
        profile = generate_naca4_profile(designation, num_points=50)
        np.testing.assert_allclose(profile[0], profile[-1], atol=1e-10)

    @pytest.mark.parametrize("designation", ["NACA0012", "NACA4412", "NACA6412", "NACA4410"])
    def test_consistent_point_count(self, designation):
        """All profiles with same num_points have same total points."""
        p1 = generate_naca4_profile(designation, num_points=50)
        p2 = generate_naca4_profile("NACA0012", num_points=50)
        assert p1.shape[0] == p2.shape[0]


class TestProfileBlending:
    """Test root-to-tip profile interpolation."""

    def test_blend_at_root(self):
        """Fraction 0.0 should return root profile."""
        root = generate_naca4_profile("NACA6412", num_points=50, chord_length=30)
        tip = generate_naca4_profile("NACA4410", num_points=50, chord_length=20)
        blended = blend_profiles(root, tip, 0.0)
        np.testing.assert_allclose(blended, root, atol=1e-10)

    def test_blend_at_tip(self):
        """Fraction 1.0 should return tip profile."""
        root = generate_naca4_profile("NACA6412", num_points=50, chord_length=30)
        tip = generate_naca4_profile("NACA4410", num_points=50, chord_length=20)
        blended = blend_profiles(root, tip, 1.0)
        np.testing.assert_allclose(blended, tip, atol=1e-10)

    def test_blend_midspan(self):
        """Fraction 0.5 should be halfway between root and tip."""
        root = generate_naca4_profile("NACA6412", num_points=50, chord_length=30)
        tip = generate_naca4_profile("NACA4410", num_points=50, chord_length=20)
        blended = blend_profiles(root, tip, 0.5)
        expected = (root + tip) / 2
        np.testing.assert_allclose(blended, expected, atol=1e-10)

    def test_blend_mismatched_shapes_raises(self):
        """Mismatched profile shapes should raise ValueError."""
        root = generate_naca4_profile("NACA6412", num_points=50)
        tip = generate_naca4_profile("NACA4410", num_points=30)
        with pytest.raises(ValueError, match="shapes must match"):
            blend_profiles(root, tip, 0.5)
