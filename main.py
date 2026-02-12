"""CLI entry point for the Ducted Fan Generator.

Usage:
    python main.py --analyze                    # Run analysis + validation report
    python main.py --generate                   # Generate STLs (with validation gate)
    python main.py --generate --part blade_ring_1  # Generate single part
    python main.py --config custom.yaml --analyze  # Use custom config
    python main.py --validate-only              # Validate existing STLs
    python main.py --view                       # Open interactive 3D viewer
    python main.py --generate --view            # Generate then view
"""

import argparse
import sys
import os

from src.config import load_config, ConfigError
from src.bemt import BEMTSolver
from src.assembly import AssemblyGenerator
from validation.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Parametric Multi-Stage Counter-Rotating Ducted Fan Generator"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file (default: config/default.yaml)"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run BEMT analysis and validation, output report"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate STL files (gated by validation)"
    )
    parser.add_argument(
        "--part", type=str, default=None,
        help="Generate only a specific part (e.g., blade_ring_stage_1)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Run validators on existing STLs"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Generate STLs even if validation fails (use with caution)"
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Open interactive 3D viewer for STL files in output/"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Open Qt-based GUI with config editor and 3D viewer"
    )

    args = parser.parse_args()

    if not any([args.analyze, args.generate, args.validate_only, args.view, args.gui]):
        parser.print_help()
        sys.exit(1)

    # Load configuration
    try:
        print("Loading configuration...")
        config = load_config(args.config)
        print(f"  Duct ID: {config['duct']['inner_diameter']}mm")
        print(f"  Hub OD:  {config['derived']['hub_od']:.1f}mm")
        print(f"  Stages:  {len(config['blades']['stages'])}")
        print(f"  RPMs:    {config['derived']['stage_rpms']}")
    except ConfigError as e:
        print(f"Configuration error:\n{e}", file=sys.stderr)
        sys.exit(1)

    # BEMT analysis (needed for both analyze and generate)
    print("\nRunning BEMT analysis...")
    solver = BEMTSolver(config)
    bemt_results = solver.solve_all_stages()
    print(f"  Total thrust: {bemt_results.total_thrust:.2f} N")
    print(f"  Total power:  {bemt_results.total_power:.2f} W")
    for stage in bemt_results.stages:
        print(f"  Stage {stage.stage_index}: {stage.rpm:.0f} RPM, "
              f"T={stage.total_thrust:.2f}N, Q={stage.total_torque:.4f}N-m")

    if args.analyze:
        run_analysis(config, bemt_results)
    if args.generate:
        run_generate(config, bemt_results, args.part, args.force)
    if args.validate_only:
        run_validate_only(config)
    if args.view:
        run_view()
    if args.gui:
        run_gui(args.config)


def run_analysis(config, bemt_results):
    """Step 1: Analysis + validation report."""
    print("\n" + "=" * 60)
    print("RUNNING FULL ANALYSIS AND VALIDATION")
    print("=" * 60)

    generator = AssemblyGenerator(config, bemt_results)
    analysis = generator.run_analysis()

    # Generate report
    report_gen = ReportGenerator(config)
    report_text = report_gen.generate_validation_report(analysis)
    bom_text = report_gen.generate_bom()

    print(report_text)
    print(f"\nReport saved to: output/validation_report.txt")
    print(f"BOM saved to: output/bom.txt")

    if analysis["all_passed"]:
        print("\n*** ALL VALIDATIONS PASSED ***")
        print("Ready for geometry generation: python main.py --generate")
    else:
        print(f"\n*** {len(analysis['failures'])} VALIDATION FAILURE(S) ***")
        print("Review report and adjust config before generating geometry.")
        print("Use --force to override (not recommended).")

    return analysis


def run_generate(config, bemt_results, part=None, force=False):
    """Step 2: Generate STLs with validation gate."""
    print("\n" + "=" * 60)
    print("GENERATING CAD GEOMETRY")
    print("=" * 60)

    generator = AssemblyGenerator(config, bemt_results)

    # Run analysis first
    print("Running pre-generation validation...")
    analysis = generator.run_analysis()

    if not analysis["all_passed"] and not force:
        print("\nValidation FAILED. Cannot generate geometry.")
        print("Failures:")
        for f in analysis["failures"]:
            print(f"  - {f}")
        print("\nUse --force to override (not recommended)")
        print("Or run --analyze for full report with suggestions")
        sys.exit(1)
    elif not analysis["all_passed"] and force:
        print("\nWARNING: Validation failed but --force specified. Proceeding...")

    # Generate and export
    parts_filter = [part] if part else None
    print("\nGenerating geometry...")
    result = generator.generate_and_export(parts_filter)

    print(f"\nExported {len(result['exported_files'])} STL files:")
    for f in result["exported_files"]:
        print(f"  {os.path.basename(f)}")

    # Report mesh validation
    mesh_pass = sum(1 for r in result["mesh_results"] if r.passed)
    mesh_total = len(result["mesh_results"])
    print(f"\nMesh validation: {mesh_pass}/{mesh_total} passed")

    # Report assembly validation
    asm_pass = sum(1 for r in result["assembly_results"] if r.passed)
    asm_total = len(result["assembly_results"])
    print(f"Assembly validation: {asm_pass}/{asm_total} passed")

    # Generate final report
    report_gen = ReportGenerator(config)
    report_gen.generate_validation_report(analysis)
    report_gen.generate_bom()
    print("\nReports saved to output/")


def run_validate_only(config):
    """Validate existing STL files."""
    from validation.mesh_validator import MeshValidator
    import trimesh

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        print("No output directory found. Generate STLs first.")
        sys.exit(1)

    validator = MeshValidator(config)
    stl_files = [f for f in os.listdir(output_dir) if f.endswith(".stl")]

    if not stl_files:
        print("No STL files found in output/")
        sys.exit(1)

    print(f"Validating {len(stl_files)} STL files...")
    for stl_file in sorted(stl_files):
        filepath = os.path.join(output_dir, stl_file)
        result = validator.validate_stl_file(filepath, stl_file)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {stl_file}: "
              f"{'watertight' if result.is_watertight else 'NOT watertight'}, "
              f"vol={result.volume_mm3:.0f}mm³, "
              f"bb={result.bounding_box[0]:.1f}x{result.bounding_box[1]:.1f}x{result.bounding_box[2]:.1f}mm")


def run_view():
    """Open interactive 3D viewer for generated STL files."""
    from src.viewer import view_assembly

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        print("No output directory found. Generate STLs first with --generate.")
        sys.exit(1)

    stl_files = [f for f in os.listdir(output_dir) if f.endswith(".stl")]
    if not stl_files:
        print("No STL files found in output/. Generate STLs first with --generate.")
        sys.exit(1)

    print(f"\nOpening viewer with {len(stl_files)} STL files...")
    view_assembly(output_dir)


def run_gui(config_path=None):
    """Open Qt-based GUI with config editor and 3D viewer."""
    from src.qt_viewer import launch_gui

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    launch_gui(config_path=config_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
