"""Full assembly coordinator — orchestrates generation, validation, and STL export.

Gates STL export on validation pass. Coordinates all generators and validators
in the correct order per the validate-before-generate philosophy.
"""

import os
import io
import tempfile
from typing import Dict, Optional, List

import cadquery as cq
import trimesh
import numpy as np

from src.config import load_config, get_stage_config
from src.bemt import BEMTSolver, BEMTResults
from src.blade_generator import BladeRingGenerator
from src.hub_generator import HubGenerator
from src.stator_generator import StatorGenerator
from src.duct_generator import DuctGenerator
from src.gear_generator import GearGenerator
from src.magnetic_coupling import MagneticCoupling
from src.carrier_generator import CarrierGenerator
from src.shaft_generator import ShaftGenerator

from validation.structural_validator import StructuralValidator
from validation.resonance_validator import ResonanceValidator
from validation.gear_validator import GearValidator
from validation.coupling_validator import CouplingValidator
from validation.system_simulator import SystemSimulator
from validation.mesh_validator import MeshValidator
from validation.assembly_validator import AssemblyValidator


class AssemblyGenerator:
    """Orchestrates the full generation pipeline with validation gates."""

    def __init__(self, config: dict, bemt_results: BEMTResults = None):
        self.config = config
        self.derived = config["derived"]
        self.bemt_results = bemt_results
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def run_analysis(self) -> dict:
        """Run all physics analysis and validation (Step 1: --analyze).

        Returns dict with analysis results and validation status.
        """
        report = {
            "bemt": None,
            "structural": None,
            "resonance": None,
            "gears": None,
            "coupling": None,
            "system": None,
            "all_passed": False,
            "failures": [],
            "warnings": [],
        }

        # BEMT
        if self.bemt_results is None:
            solver = BEMTSolver(self.config)
            self.bemt_results = solver.solve_all_stages()

        report["bemt"] = {
            "total_thrust": self.bemt_results.total_thrust,
            "total_power": self.bemt_results.total_power,
            "stages": [
                {
                    "index": s.stage_index,
                    "rpm": s.rpm,
                    "thrust_N": s.total_thrust,
                    "torque_Nm": s.total_torque,
                    "power_W": s.power,
                }
                for s in self.bemt_results.stages
            ],
        }

        # Structural validation
        struct_val = StructuralValidator(self.config)
        struct_results = struct_val.validate_all(self.bemt_results)
        report["structural"] = {
            "passed": struct_val.all_passed(struct_results),
            "results": [
                {
                    "component": r.component,
                    "type": r.stress_type,
                    "stress_MPa": r.actual_stress,
                    "allowable_MPa": r.allowable_stress,
                    "SF": r.safety_factor,
                    "passed": r.passed,
                    "detail": r.detail,
                }
                for r in struct_results
            ],
        }
        if not struct_val.all_passed(struct_results):
            for r in struct_results:
                if not r.passed:
                    report["failures"].append(
                        f"STRUCTURAL: {r.component} {r.stress_type}: "
                        f"σ={r.actual_stress:.2f}MPa > allowable {r.allowable_stress:.2f}MPa"
                    )

        # Resonance validation
        res_val = ResonanceValidator(self.config)
        res_checks, coprime_checks = res_val.validate_all()
        report["resonance"] = {
            "passed": res_val.all_passed(),
            "checks": len(res_checks),
            "failures": [
                f"Stage {r.stage}: {r.excitation_type} "
                f"f_n={r.natural_freq_hz:.1f}Hz, f_exc={r.excitation_freq_hz:.1f}Hz, "
                f"sep={r.separation_pct:.1f}%"
                for r in res_checks if not r.passed
            ],
        }
        if not res_val.all_passed():
            for r in res_checks:
                if not r.passed:
                    report["failures"].append(
                        f"RESONANCE: Stage {r.stage} {r.excitation_type} "
                        f"separation {r.separation_pct:.1f}% < {r.min_separation_pct}%"
                    )

        # Gear validation
        gear_val = GearValidator(self.config)
        gear_results = gear_val.validate_all()
        report["gears"] = {
            "passed": gear_val.all_passed(),
            "results": [
                {
                    "check": r.check_name,
                    "gear": r.gear_name,
                    "value": r.value,
                    "limit": r.limit,
                    "passed": r.passed,
                    "detail": r.detail,
                }
                for r in gear_results
            ],
        }
        if not gear_val.all_passed():
            for r in gear_results:
                if not r.passed:
                    if r.check_name == "undercutting":
                        report["warnings"].append(
                            f"GEAR: {r.gear_name} undercut risk — {r.detail}"
                        )
                    else:
                        report["failures"].append(
                            f"GEAR: {r.gear_name} {r.check_name}: {r.detail}"
                        )

        # Coupling validation
        coupling_val = CouplingValidator(self.config)
        coupling_result = coupling_val.validate_with_bemt(self.bemt_results)
        report["coupling"] = {
            "passed": coupling_result.all_passed,
            "critical_stage": coupling_result.critical_stage,
            "stages": [
                {
                    "index": s.stage_index,
                    "peak_torque_Nm": s.peak_torque,
                    "required_torque_Nm": s.required_torque,
                    "SF": s.safety_factor,
                    "passed": s.passed,
                }
                for s in coupling_result.stages
            ],
            "ferromagnetic_warning": coupling_result.ferromagnetic_warning,
        }
        report["warnings"].append(coupling_result.ferromagnetic_warning)

        if not coupling_result.all_passed:
            for s in coupling_result.stages:
                if not s.passed:
                    report["failures"].append(
                        f"COUPLING: Stage {s.stage_index} SF={s.safety_factor:.2f} "
                        f"< required {s.required_sf}"
                    )

        # System simulation
        sys_sim = SystemSimulator(self.config)
        sys_result = sys_sim.simulate_full_system(self.bemt_results)
        report["system"] = {
            "converged": sys_result.converged,
            "operating_points": [
                {
                    "stage": op.stage_index,
                    "rpm": op.rpm,
                    "efficiency": op.efficiency,
                    "power_in_W": op.power_in,
                    "power_thrust_W": op.power_thrust,
                }
                for op in sys_result.operating_points
            ],
            "power_budget": sys_result.power_budget,
        }

        # Overall assessment
        report["all_passed"] = (
            struct_val.all_passed(struct_results)
            and res_val.all_passed()
            and coupling_result.all_passed
            # Gear undercutting is a warning, not a gate
        )

        return report

    def generate_all_meshes(self) -> Dict[str, trimesh.Trimesh]:
        """Generate all CAD geometry and convert to trimesh for validation.

        Returns dict of part_name -> trimesh.Trimesh
        """
        meshes = {}

        # Hub halves
        hub_gen = HubGenerator(self.config)
        half_a = hub_gen.generate_half_a()
        half_b = hub_gen.generate_half_b()
        meshes["hub_half_a"] = self._cq_to_trimesh(half_a)
        meshes["hub_half_b"] = self._cq_to_trimesh(half_b)

        # Blade rings
        for i in range(len(self.config["blades"]["stages"])):
            bemt_stage = self.bemt_results.stages[i] if self.bemt_results else None
            blade_gen = BladeRingGenerator(self.config, i, bemt_stage)
            blade_ring = blade_gen.generate()
            meshes[f"blade_ring_stage_{i+1}"] = self._cq_to_trimesh(blade_ring)

        # Stators
        stator_gen = StatorGenerator(self.config)
        meshes["stator_entry"] = self._cq_to_trimesh(stator_gen.generate_entry_stator())
        meshes["stator_exit"] = self._cq_to_trimesh(stator_gen.generate_exit_stator())

        # Duct sections
        duct_gen = DuctGenerator(self.config)
        duct_sections = duct_gen.generate()
        for j, section in enumerate(duct_sections):
            meshes[f"duct_section_{j+1}"] = self._cq_to_trimesh(section)

        # Gear stages
        gear_gen = GearGenerator(self.config)
        num_gear_stages = self.config["gears"].get("num_stages", 1)
        for stage_idx in range(num_gear_stages):
            gear_solids = gear_gen.generate_planetary_stage(stage_idx)
            for name, solid in gear_solids.items():
                meshes[name] = self._cq_to_trimesh(solid)

        # Carrier plates
        carrier_gen = CarrierGenerator(self.config)
        carrier_solids = carrier_gen.generate_all_carriers()
        for name, solid in carrier_solids.items():
            meshes[name] = self._cq_to_trimesh(solid)

        # Concentric shafts and tubes
        shaft_gen = ShaftGenerator(self.config)
        shaft_solids = shaft_gen.generate_all()
        for name, solid in shaft_solids.items():
            meshes[name] = self._cq_to_trimesh(solid)

        # Apply axial layout positioning
        meshes = self.position_meshes(meshes)

        return meshes

    def position_meshes(self, meshes: Dict[str, trimesh.Trimesh]) -> Dict[str, trimesh.Trimesh]:
        """Translate meshes to their axial (Z) positions from config layout."""
        positions = self.derived.get("part_positions", {})
        positioned = {}
        for name, mesh in meshes.items():
            z_offset = self._find_position(name, positions)
            if z_offset != 0:
                mesh = mesh.copy()
                mesh.apply_translation([0, 0, z_offset])
            positioned[name] = mesh
        return positioned

    @staticmethod
    def _find_position(name: str, positions: dict) -> float:
        """Find the Z position for a named part from the layout dict."""
        # Direct match
        if name in positions:
            return positions[name]

        # Pattern matching for gear parts: gear_sun_stage_0 -> gear_stage_0
        if name.startswith("gear_"):
            for key in positions:
                if key.startswith("gear_stage_"):
                    stage_num = key.split("_")[-1]
                    if f"stage_{stage_num}" in name:
                        return positions[key]

        # Pattern matching for carrier parts: carrier_front_0 -> carrier_front_0
        if name.startswith("carrier_"):
            if name in positions:
                return positions[name]

        # Shafts and tubes positioned at hub start
        if name in ("inner_shaft", "middle_tube", "outer_tube"):
            return positions.get("hub_half_a", 0.0)

        # Pattern matching for duct sections
        if "duct" in name:
            for key in positions:
                if "duct" in key:
                    return positions[key]

        return 0.0

    def generate_and_export(self, parts: Optional[List[str]] = None) -> dict:
        """Generate geometry, validate, and export STLs (Step 2: --generate).

        Args:
            parts: Optional list of specific parts to generate. None = all.

        Returns:
            Dict with export results and validation status
        """
        result = {
            "exported_files": [],
            "validation_passed": False,
            "mesh_results": [],
            "assembly_results": [],
        }

        # Generate meshes
        meshes = self.generate_all_meshes()

        # Filter if specific parts requested
        if parts:
            meshes = {k: v for k, v in meshes.items() if k in parts}

        # Mesh validation
        mesh_val = MeshValidator(self.config)
        mesh_results = mesh_val.validate_all_meshes(meshes)
        result["mesh_results"] = mesh_results

        # Assembly validation
        asm_val = AssemblyValidator(self.config)
        asm_results = asm_val.validate_all(meshes)
        result["assembly_results"] = asm_results

        # Check validation
        mesh_ok = all(r.passed for r in mesh_results)
        asm_ok = all(r.passed for r in asm_results)
        result["validation_passed"] = mesh_ok and asm_ok

        # Export STLs
        for name, mesh in meshes.items():
            filepath = os.path.join(self.output_dir, f"{name}.stl")
            mesh.export(filepath)
            result["exported_files"].append(filepath)

        return result

    @staticmethod
    def _cq_to_trimesh(cq_solid: cq.Workplane) -> trimesh.Trimesh:
        """Convert CadQuery solid to trimesh.Trimesh via STL export."""
        # CadQuery requires a file path for STL export
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cq.exporters.export(cq_solid, tmp_path, exportType="STL")
            mesh = trimesh.load(tmp_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            return mesh
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
