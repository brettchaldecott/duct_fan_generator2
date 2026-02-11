"""Consolidated validation report output.

Generates human-readable text reports from analysis and validation results.
"""

import os
from datetime import datetime
from typing import Dict, List


class ReportGenerator:
    """Generates consolidated validation and analysis reports."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_validation_report(self, analysis: dict, filepath: str = None) -> str:
        """Generate validation report text.

        Args:
            analysis: Analysis results dict from AssemblyGenerator.run_analysis()
            filepath: Optional output file path

        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 70)
        lines.append("DUCTED FAN GENERATOR — VALIDATION REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Overall status
        if analysis["all_passed"]:
            lines.append("OVERALL STATUS: *** PASS ***")
            lines.append("All physics validations passed. Ready for geometry generation.")
        else:
            lines.append("OVERALL STATUS: *** FAIL ***")
            lines.append(f"  {len(analysis['failures'])} failure(s) found.")
            lines.append("  Geometry generation should NOT proceed until resolved.")
        lines.append("")

        # BEMT results
        lines.append("-" * 70)
        lines.append("BEMT ANALYSIS")
        lines.append("-" * 70)
        bemt = analysis.get("bemt", {})
        lines.append(f"  Total thrust: {bemt.get('total_thrust', 0):.2f} N")
        lines.append(f"  Total power:  {bemt.get('total_power', 0):.2f} W")
        for s in bemt.get("stages", []):
            lines.append(
                f"  Stage {s['index']}: RPM={s['rpm']:.0f}, "
                f"thrust={s['thrust_N']:.2f}N, torque={s['torque_Nm']:.4f}N-m, "
                f"power={s['power_W']:.2f}W"
            )
        lines.append("")

        # Structural
        lines.append("-" * 70)
        lines.append("STRUCTURAL VALIDATION")
        lines.append("-" * 70)
        struct = analysis.get("structural", {})
        for r in struct.get("results", []):
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"  [{status}] {r['component']} ({r['type']}): "
                f"σ={r['stress_MPa']:.2f}MPa, allowable={r['allowable_MPa']:.2f}MPa, "
                f"SF={r['SF']:.2f}"
            )
            if not r["passed"]:
                lines.append(f"         → {r['detail']}")
        lines.append("")

        # Resonance
        lines.append("-" * 70)
        lines.append("RESONANCE VALIDATION")
        lines.append("-" * 70)
        res = analysis.get("resonance", {})
        if res.get("passed"):
            lines.append("  All resonance checks PASSED")
        else:
            for f in res.get("failures", []):
                lines.append(f"  [FAIL] {f}")
        lines.append("")

        # Gears
        lines.append("-" * 70)
        lines.append("GEAR VALIDATION")
        lines.append("-" * 70)
        gears = analysis.get("gears", {})
        for r in gears.get("results", []):
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f"  [{status}] {r['gear']} {r['check']}: {r['detail']}")
        lines.append("")

        # Coupling
        lines.append("-" * 70)
        lines.append("MAGNETIC COUPLING VALIDATION")
        lines.append("-" * 70)
        coupling = analysis.get("coupling", {})
        for s in coupling.get("stages", []):
            status = "PASS" if s["passed"] else "FAIL"
            lines.append(
                f"  [{status}] Stage {s['index']}: peak={s['peak_torque_Nm']:.4f}N-m, "
                f"required={s['required_torque_Nm']:.4f}N-m, SF={s['SF']:.2f}"
            )
        if coupling.get("ferromagnetic_warning"):
            lines.append(f"  NOTE: {coupling['ferromagnetic_warning']}")
        lines.append("")

        # System simulation
        lines.append("-" * 70)
        lines.append("SYSTEM SIMULATION")
        lines.append("-" * 70)
        sys_data = analysis.get("system", {})
        lines.append(f"  Converged: {sys_data.get('converged', False)}")
        for op in sys_data.get("operating_points", []):
            lines.append(
                f"  Stage {op['stage']}: RPM={op['rpm']:.0f}, "
                f"η={op['efficiency']:.3f}, "
                f"P_in={op['power_in_W']:.1f}W, P_thrust={op['power_thrust_W']:.1f}W"
            )
        pb = sys_data.get("power_budget", {})
        if pb:
            lines.append(
                f"  Power budget: P_in={pb.get('total_power_in_W', 0):.1f}W, "
                f"P_thrust={pb.get('total_power_thrust_W', 0):.1f}W, "
                f"η_overall={pb.get('overall_efficiency', 0):.3f}"
            )
        lines.append("")

        # Failures summary
        if analysis["failures"]:
            lines.append("-" * 70)
            lines.append("FAILURES — ACTION REQUIRED")
            lines.append("-" * 70)
            for f in analysis["failures"]:
                lines.append(f"  ✗ {f}")
            lines.append("")
            lines.append("SUGGESTIONS:")
            self._add_suggestions(lines, analysis)
            lines.append("")

        # Warnings
        if analysis["warnings"]:
            lines.append("-" * 70)
            lines.append("WARNINGS")
            lines.append("-" * 70)
            for w in analysis["warnings"]:
                lines.append(f"  ! {w}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        if filepath is None:
            filepath = os.path.join(self.output_dir, "validation_report.txt")
        with open(filepath, "w") as f:
            f.write(report_text)

        return report_text

    def generate_bom(self, filepath: str = None) -> str:
        """Generate bill of materials."""
        lines = []
        lines.append("=" * 50)
        lines.append("BILL OF MATERIALS")
        lines.append("=" * 50)
        lines.append("")

        # Bearings
        lines.append("BEARINGS:")
        for name, bearing in self.config["bearings"].items():
            qty = 2  # typically 2 per bearing type
            lines.append(
                f"  {bearing['name']} ({bearing['id']}x{bearing['od']}x{bearing['width']}mm) — qty: {qty}"
            )
        lines.append("")

        # Magnets
        lines.append("MAGNETS (N52 NdFeB):")
        for i, stage in enumerate(self.config["magnetic_coupling"]["stages"]):
            n_magnets = stage["num_pole_pairs"] * 2 * 2  # inner + outer ring
            lines.append(
                f"  Stage {i+1}: {stage['magnet_diameter']}mm dia × "
                f"{stage['magnet_thickness']}mm thick — qty: {n_magnets}"
            )
        lines.append("")

        # Fasteners
        lines.append("FASTENERS:")
        lines.append(f"  M3×12 socket head cap screws — qty: 12 (hub assembly)")
        lines.append(f"  M{self.config['motor']['mounting_holes_diameter']:.0f} motor mounting — "
                     f"qty: {self.config['motor']['mounting_holes_count']}")
        lines.append("")

        # Motor
        lines.append("MOTOR:")
        lines.append(f"  DC motor: {self.config['motor']['body_diameter']}mm × "
                     f"{self.config['motor']['body_length']}mm, "
                     f"{self.config['motor']['rpm']} RPM @ {self.config['motor']['voltage']}V")

        bom_text = "\n".join(lines)

        if filepath is None:
            filepath = os.path.join(self.output_dir, "bom.txt")
        with open(filepath, "w") as f:
            f.write(bom_text)

        return bom_text

    def _add_suggestions(self, lines: List[str], analysis: dict) -> None:
        """Add failure-specific suggestions to the report."""
        for failure in analysis["failures"]:
            if "centrifugal" in failure.lower() and "stage_1" in failure.lower():
                lines.append(
                    "  → Reduce Stage 1 RPM or duct diameter to lower centrifugal stress"
                )
                lines.append(
                    "  → Consider switching blade material to a higher-UTS composite"
                )
            elif "planet" in failure.lower() and "bending" in failure.lower():
                lines.append(
                    "  → Increase planet tooth count (increase gear module or change ratio)"
                )
                lines.append(
                    "  → Switch to 25° pressure angle to reduce bending stress"
                )
            elif "hoop" in failure.lower() or "hub_wall" in failure.lower():
                lines.append(
                    "  → Increase hub wall thickness at coupling zones"
                )
                lines.append(
                    "  → Reduce number of magnet pole pairs or magnet size"
                )
            elif "coupling" in failure.lower():
                lines.append(
                    "  → Increase magnet thickness for the failing stage"
                )
                lines.append(
                    "  → Reduce wall thickness between magnet rings (if structurally safe)"
                )
            elif "resonance" in failure.lower():
                lines.append(
                    "  → Adjust blade count or operating RPM to move away from resonance"
                )
