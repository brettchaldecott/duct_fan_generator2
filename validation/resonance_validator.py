"""Resonance validation — vibration safety checks.

Checks blade natural frequencies vs excitation frequencies to ensure
no resonance at operating RPMs. Generates Campbell diagram data.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ResonanceCheck:
    """Result of a single resonance check."""
    stage: int
    natural_freq_hz: float
    excitation_type: str   # "BPF", "2xBPF", "3xBPF", "shaft"
    excitation_freq_hz: float
    separation_pct: float  # % separation between frequencies
    min_separation_pct: float  # minimum required (15%)
    passed: bool
    detail: str = ""


@dataclass
class CoprimCheck:
    """Result of coprime blade count check between stages."""
    stage_a: int
    stage_b: int
    blades_a: int
    blades_b: int
    gcd: int
    passed: bool


class ResonanceValidator:
    """Validates that no resonance conditions exist at operating points."""

    MIN_SEPARATION_PCT = 15.0  # ±15% separation required

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]

    def validate_all(self) -> Tuple[List[ResonanceCheck], List[CoprimCheck]]:
        """Run all resonance validations."""
        resonance_checks = []
        for i in range(len(self.config["blades"]["stages"])):
            resonance_checks.extend(self.check_stage_resonance(i))

        coprime_checks = self.check_coprime_blade_counts()

        return resonance_checks, coprime_checks

    def blade_natural_frequency(self, stage_index: int) -> float:
        """Estimate first natural frequency of blade using Euler-Bernoulli cantilever model.

        f_n = (λ²/2π) × √(EI / (ρAL⁴))

        For first mode of cantilever: λ₁ = 1.875
        """
        blade_cfg = self.config["blades"]["stages"][stage_index]
        mat = self.config["materials"]["blade"]

        # Blade span (cantilever length)
        span = self.derived["blade_span"] / 1000  # mm to m

        # Approximate blade cross-section as rectangular
        # Use mid-span chord estimate (average of typical BEMT results)
        chord_est = 0.03  # ~30mm estimated chord
        thickness_est = chord_est * 0.12  # ~12% thickness ratio

        # Section properties
        E = mat["elastic_modulus"] * 1e6  # MPa to Pa
        rho = mat["density"]  # kg/m³
        b = chord_est  # width
        h = thickness_est  # height

        I = b * h**3 / 12  # second moment of area
        A = b * h  # cross-sectional area

        # First mode cantilever eigenvalue
        lambda_1 = 1.875

        # Natural frequency
        f_n = (lambda_1**2 / (2 * math.pi)) * math.sqrt(E * I / (rho * A * span**4))

        return f_n

    def check_stage_resonance(self, stage_index: int) -> List[ResonanceCheck]:
        """Check resonance conditions for a single stage."""
        results = []
        rpm = self.derived["stage_rpms"][stage_index]
        n_blades = self.config["blades"]["stages"][stage_index]["num_blades"]

        # Natural frequency
        f_n = self.blade_natural_frequency(stage_index)

        # Excitation frequencies
        shaft_freq = rpm / 60  # Hz
        bpf = n_blades * rpm / 60  # Blade Passing Frequency

        excitations = [
            ("shaft", shaft_freq),
            ("BPF", bpf),
            ("2xBPF", 2 * bpf),
            ("3xBPF", 3 * bpf),
        ]

        for exc_type, exc_freq in excitations:
            if exc_freq > 0:
                separation = abs(f_n - exc_freq) / exc_freq * 100
            else:
                separation = float("inf")

            passed = separation >= self.MIN_SEPARATION_PCT

            results.append(ResonanceCheck(
                stage=stage_index,
                natural_freq_hz=f_n,
                excitation_type=exc_type,
                excitation_freq_hz=exc_freq,
                separation_pct=separation,
                min_separation_pct=self.MIN_SEPARATION_PCT,
                passed=passed,
                detail=f"f_n={f_n:.1f}Hz, {exc_type}={exc_freq:.1f}Hz"
            ))

        return results

    def check_coprime_blade_counts(self) -> List[CoprimCheck]:
        """Check that blade counts between adjacent stages are coprime."""
        results = []
        stages = self.config["blades"]["stages"]

        for i in range(len(stages) - 1):
            n_a = stages[i]["num_blades"]
            n_b = stages[i + 1]["num_blades"]
            gcd = math.gcd(n_a, n_b)

            results.append(CoprimCheck(
                stage_a=i,
                stage_b=i + 1,
                blades_a=n_a,
                blades_b=n_b,
                gcd=gcd,
                passed=gcd == 1,
            ))

        return results

    def generate_campbell_data(self, rpm_range=None) -> dict:
        """Generate Campbell diagram data for visualization.

        Returns dict with:
            rpms: array of RPM values
            natural_freqs: dict of stage -> frequency
            excitation_lines: dict of label -> array of frequencies
        """
        if rpm_range is None:
            max_rpm = max(self.derived["stage_rpms"]) * 1.2
            rpm_range = np.linspace(0, max_rpm, 200)

        data = {
            "rpms": rpm_range,
            "natural_freqs": {},
            "excitation_lines": {},
        }

        for i, blade_cfg in enumerate(self.config["blades"]["stages"]):
            f_n = self.blade_natural_frequency(i)
            data["natural_freqs"][f"stage_{i+1}"] = f_n

            n_blades = blade_cfg["num_blades"]
            data["excitation_lines"][f"stage_{i+1}_shaft"] = rpm_range / 60
            data["excitation_lines"][f"stage_{i+1}_BPF"] = n_blades * rpm_range / 60
            data["excitation_lines"][f"stage_{i+1}_2xBPF"] = 2 * n_blades * rpm_range / 60

        return data

    def all_passed(self) -> bool:
        """Check if all resonance validations passed."""
        resonance_checks, coprime_checks = self.validate_all()
        return (
            all(r.passed for r in resonance_checks) and
            all(c.passed for c in coprime_checks)
        )
