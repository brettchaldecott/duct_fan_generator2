"""Tests for resonance validation."""

import math
import pytest
from validation.resonance_validator import ResonanceValidator


class TestResonanceChecks:
    """Test resonance frequency separation."""

    def test_resonance_checks_run(self, default_config):
        """Resonance checks run for all stages and excitation types."""
        validator = ResonanceValidator(default_config)
        resonance_checks, _ = validator.validate_all()
        # Should have checks for shaft, BPF, 2xBPF, 3xBPF per stage
        num_stages = len(default_config["blades"]["stages"])
        assert len(resonance_checks) == num_stages * 4

    def test_no_bpf_resonance(self, default_config):
        """No blade natural frequency within ±15% of BPF at any stage."""
        validator = ResonanceValidator(default_config)
        resonance_checks, _ = validator.validate_all()
        for check in resonance_checks:
            if check.excitation_type == "BPF":
                assert check.passed, (
                    f"Stage {check.stage}: BPF resonance! "
                    f"f_n={check.natural_freq_hz:.1f}Hz, "
                    f"BPF={check.excitation_freq_hz:.1f}Hz, "
                    f"separation={check.separation_pct:.1f}%"
                )

    def test_natural_frequencies_positive(self, default_config):
        """All computed natural frequencies should be positive."""
        validator = ResonanceValidator(default_config)
        resonance_checks, _ = validator.validate_all()
        for check in resonance_checks:
            assert check.natural_freq_hz > 0

    def test_natural_frequency_reasonable(self, default_config):
        """Natural frequency should be in a physically reasonable range."""
        validator = ResonanceValidator(default_config)
        for i in range(len(default_config["blades"]["stages"])):
            f_n = validator.blade_natural_frequency(i)
            # For a ~100mm blade span, first mode should be 10-50000 Hz
            assert 10 < f_n < 50000, f"Natural frequency {f_n}Hz seems unreasonable"

    def test_separation_percentage_calculated(self, default_config):
        """All checks report separation percentage."""
        validator = ResonanceValidator(default_config)
        resonance_checks, _ = validator.validate_all()
        for check in resonance_checks:
            assert check.separation_pct >= 0


class TestCoprimeBlades:
    """Test coprime blade count checks."""

    def test_adjacent_stages_coprime(self, default_config):
        """Blade counts between adjacent stages share no common factor > 1."""
        validator = ResonanceValidator(default_config)
        _, coprime_checks = validator.validate_all()
        for check in coprime_checks:
            assert check.passed, (
                f"Stages {check.stage_a}-{check.stage_b}: "
                f"blades {check.blades_a},{check.blades_b} "
                f"have GCD={check.gcd}"
            )

    def test_default_blade_counts_are_coprime(self, default_config):
        """Default blade counts (7, 9, 11) are pairwise coprime."""
        stages = default_config["blades"]["stages"]
        counts = [s["num_blades"] for s in stages]
        for i in range(len(counts)):
            for j in range(i + 1, len(counts)):
                assert math.gcd(counts[i], counts[j]) == 1, (
                    f"Blade counts {counts[i]} and {counts[j]} are not coprime"
                )


class TestCampbellDiagram:
    """Test Campbell diagram data generation."""

    def test_campbell_data_generates(self, default_config):
        """Campbell diagram data generates without error."""
        validator = ResonanceValidator(default_config)
        data = validator.generate_campbell_data()
        assert "rpms" in data
        assert "natural_freqs" in data
        assert "excitation_lines" in data

    def test_campbell_has_all_stages(self, default_config):
        """Campbell data includes entries for all stages."""
        validator = ResonanceValidator(default_config)
        data = validator.generate_campbell_data()
        num_stages = len(default_config["blades"]["stages"])
        assert len(data["natural_freqs"]) == num_stages

    def test_campbell_excitation_lines_increase(self, default_config):
        """Excitation frequency lines should increase with RPM."""
        validator = ResonanceValidator(default_config)
        data = validator.generate_campbell_data()
        for key, freqs in data["excitation_lines"].items():
            # Frequency should be monotonically increasing with RPM
            assert all(freqs[i] <= freqs[i+1] for i in range(len(freqs)-1))
