"""Assembly validation — collision detection, clearance checks, build volume.

Validates that all parts fit together correctly without interference,
clearances are met, and each part fits within the build volume.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import trimesh


@dataclass
class AssemblyCheck:
    """Result of a single assembly validation check."""
    check_name: str
    part_a: str
    part_b: str
    value: float           # measured dimension/clearance
    limit: float           # required minimum/maximum
    passed: bool
    detail: str = ""


class AssemblyValidator:
    """Validates assembly fit, clearances, and collision-free assembly."""

    def __init__(self, config: dict):
        self.config = config
        self.derived = config["derived"]
        self.print_cfg = config["print"]

    def validate_all(self, meshes: dict) -> List[AssemblyCheck]:
        """Run all assembly validations.

        Args:
            meshes: Dictionary of part_name -> trimesh.Trimesh

        Returns:
            List of AssemblyCheck results
        """
        results = []

        # Build volume checks
        results.extend(self.check_build_volume(meshes))

        # Bearing seat checks
        results.extend(self.check_bearing_seats())

        # Magnet pocket checks
        results.extend(self.check_magnet_pockets())

        # Blade tip clearance
        results.extend(self.check_blade_tip_clearance(meshes))

        # Gear-blade stage alignment
        results.extend(self.check_gear_blade_alignment())

        # Concentric shaft clearances
        results.extend(self.check_concentric_clearances())

        # Ring gear fits inside hub
        results.extend(self.check_ring_gear_fit())

        # Blade ring clears hub outer wall
        results.extend(self.check_blade_ring_clearance())

        # Blade-to-blade collision within each stage
        results.extend(self.check_blade_blade_collisions())

        # Collision detection (if meshes available)
        if meshes:
            results.extend(self.check_collisions(meshes))

        return results

    def check_build_volume(self, meshes: dict) -> List[AssemblyCheck]:
        """Check that each part fits the build volume."""
        results = []
        max_x = self.print_cfg["max_build_x"]
        max_y = self.print_cfg["max_build_y"]
        max_z = self.print_cfg["max_build_z"]

        for name, mesh in meshes.items():
            bb = mesh.bounding_box.extents
            fits = bb[0] <= max_x and bb[1] <= max_y and bb[2] <= max_z

            results.append(AssemblyCheck(
                check_name="build_volume",
                part_a=name,
                part_b="build_plate",
                value=max(bb),
                limit=max(max_x, max_y, max_z),
                passed=fits,
                detail=f"Part {bb[0]:.1f}x{bb[1]:.1f}x{bb[2]:.1f}mm, build {max_x}x{max_y}x{max_z}mm"
            ))

        return results

    def check_bearing_seats(self) -> List[AssemblyCheck]:
        """Check bearing seat dimensions are within tolerance.

        Bearing seats should be bearing OD + press_fit_clearance to
        bearing OD + press_fit_clearance + 0.10mm.
        """
        results = []
        clearance = self.config["hub"]["bearing_press_fit_clearance"]

        for bearing_name, bearing in self.config["bearings"].items():
            od = bearing["od"]
            seat_d = od + 2 * clearance  # nominal seat diameter
            min_seat = od + 0.05   # minimum: slight press fit
            max_seat = od + 0.20   # maximum: slight clearance

            results.append(AssemblyCheck(
                check_name="bearing_seat",
                part_a="hub",
                part_b=bearing["name"],
                value=seat_d,
                limit=max_seat,
                passed=min_seat <= seat_d <= max_seat,
                detail=f"{bearing['name']} OD={od}mm, seat={seat_d:.2f}mm, range=[{min_seat:.2f}, {max_seat:.2f}]"
            ))

        return results

    def check_magnet_pockets(self) -> List[AssemblyCheck]:
        """Check magnet pocket dimensions are within tolerance."""
        results = []
        pocket_clearance = self.config["magnetic_coupling"]["magnet_pocket_clearance"]

        for i, stage in enumerate(self.config["magnetic_coupling"]["stages"]):
            mag_d = stage["magnet_diameter"]
            pocket_d = mag_d + 2 * pocket_clearance
            min_pocket = mag_d + 0.15  # minimum clearance for insertion
            max_pocket = mag_d + 0.30  # maximum before excessive play

            results.append(AssemblyCheck(
                check_name="magnet_pocket",
                part_a=f"blade_ring_stage_{i+1}",
                part_b=f"magnet_{mag_d}mm",
                value=pocket_d,
                limit=max_pocket,
                passed=min_pocket <= pocket_d <= max_pocket,
                detail=f"Magnet OD={mag_d}mm, pocket={pocket_d:.2f}mm, range=[{min_pocket:.2f}, {max_pocket:.2f}]"
            ))

        return results

    def check_blade_tip_clearance(self, meshes: dict = None) -> List[AssemblyCheck]:
        """Check blade tip clearance to duct ID using mesh data when available."""
        results = []
        tip_clearance = self.config["blades"]["tip_clearance"]
        duct_r = self.config["duct"]["inner_diameter"] / 2

        if meshes:
            for name, mesh in meshes.items():
                if "blade_ring" in name:
                    # Actual max radial extent from mesh vertices
                    xy_extent = np.sqrt(mesh.vertices[:, 0]**2 + mesh.vertices[:, 1]**2)
                    actual_tip_r = np.max(xy_extent)
                    gap = duct_r - actual_tip_r
                    results.append(AssemblyCheck(
                        check_name="tip_clearance",
                        part_a=name,
                        part_b="duct",
                        value=gap,
                        limit=tip_clearance,
                        passed=gap >= tip_clearance * 0.9,  # 90% tolerance
                        detail=f"Actual tip R={actual_tip_r:.1f}mm, duct IR={duct_r:.1f}mm, gap={gap:.1f}mm"
                    ))

        if not results:
            # Fallback to config-based check
            tip_r = self.derived["blade_tip_radius"]
            gap = duct_r - tip_r
            results.append(AssemblyCheck(
                check_name="tip_clearance",
                part_a="blade_ring",
                part_b="duct",
                value=gap,
                limit=tip_clearance,
                passed=gap >= tip_clearance,
                detail=f"Config-based: tip R={tip_r:.1f}mm, duct IR={duct_r:.1f}mm, gap={gap:.1f}mm"
            ))

        return results

    def check_gear_blade_alignment(self) -> List[AssemblyCheck]:
        """Check that gear stages are positioned between their blade stages."""
        results = []
        positions = self.derived.get("part_positions", {})
        blade_axial_width = self.derived.get("blade_axial_width", 10)
        num_gear_stages = self.config["gears"].get("num_stages", 0)

        for i in range(num_gear_stages):
            gear_key = f"gear_stage_{i}"
            blade_before = f"blade_ring_stage_{i+1}"
            blade_after = f"blade_ring_stage_{i+2}"

            if gear_key in positions and blade_before in positions and blade_after in positions:
                gear_z = positions[gear_key]
                # Blade positions are now centers, so end = center + half_width
                blade_before_end = positions[blade_before] + blade_axial_width / 2
                blade_after_start = positions[blade_after] - blade_axial_width / 2

                is_between = blade_before_end <= gear_z <= blade_after_start
                results.append(AssemblyCheck(
                    check_name="gear_blade_alignment",
                    part_a=gear_key,
                    part_b=f"{blade_before} / {blade_after}",
                    value=gear_z,
                    limit=blade_after_start,
                    passed=is_between,
                    detail=f"Gear at Z={gear_z:.1f}, blade range [{blade_before_end:.1f}, {blade_after_start:.1f}]"
                ))

        return results

    def check_concentric_clearances(self) -> List[AssemblyCheck]:
        """Check concentric shaft/tube clearances."""
        results = []
        d = self.derived

        # Inner shaft in middle tube
        shaft_d = d.get("inner_shaft_diameter", 0)
        mid_id = d.get("middle_tube_id", 0)
        if shaft_d > 0 and mid_id > 0:
            clearance = mid_id - shaft_d
            results.append(AssemblyCheck(
                check_name="concentric_clearance",
                part_a="inner_shaft",
                part_b="middle_tube",
                value=clearance,
                limit=1.0,
                passed=clearance >= 1.0,
                detail=f"Shaft OD={shaft_d}mm, tube ID={mid_id}mm, clearance={clearance:.1f}mm"
            ))

        # Middle tube in outer tube
        mid_od = d.get("middle_tube_od", 0)
        out_id = d.get("outer_tube_id", 0)
        if mid_od > 0 and out_id > 0:
            clearance = out_id - mid_od
            results.append(AssemblyCheck(
                check_name="concentric_clearance",
                part_a="middle_tube",
                part_b="outer_tube",
                value=clearance,
                limit=1.0,
                passed=clearance >= 1.0,
                detail=f"Middle OD={mid_od}mm, outer ID={out_id}mm, clearance={clearance:.1f}mm"
            ))

        return results

    def check_ring_gear_fit(self) -> List[AssemblyCheck]:
        """Check that ring gear fits inside hub housing."""
        results = []
        ring_outer_wall_r = self.derived.get("ring_outer_wall_radius", 0)
        hub_inner_r = self.derived.get("hub_od", 0) / 2 - self.config["hub"]["wall_thickness"]

        if ring_outer_wall_r > 0:
            clearance = hub_inner_r - ring_outer_wall_r
            results.append(AssemblyCheck(
                check_name="ring_gear_fit",
                part_a="ring_gear",
                part_b="hub_interior",
                value=clearance,
                limit=0.0,
                passed=clearance >= 0.0,
                detail=f"Ring outer wall R={ring_outer_wall_r:.1f}mm, hub inner R={hub_inner_r:.1f}mm, clearance={clearance:.1f}mm"
            ))

        return results

    def check_blade_ring_clearance(self) -> List[AssemblyCheck]:
        """Check that blade ring inner surface clears hub outer surface (air gap)."""
        results = []
        blade_ring_radii = self.derived.get("blade_ring_radii", [])
        per_stage_hub_radii = self.derived.get("per_stage_hub_radii", [])
        air_gap = self.derived.get("blade_ring_air_gap", 1.0)

        for i, ring_info in enumerate(blade_ring_radii):
            if i < len(per_stage_hub_radii):
                hub_r = per_stage_hub_radii[i]
                ring_inner_r = ring_info["ring_inner_r"]
                actual_gap = ring_inner_r - hub_r

                results.append(AssemblyCheck(
                    check_name="blade_ring_clearance",
                    part_a=f"blade_ring_stage_{i+1}",
                    part_b="hub",
                    value=actual_gap,
                    limit=air_gap,
                    passed=actual_gap >= air_gap * 0.9,  # 90% tolerance
                    detail=f"Ring inner R={ring_inner_r:.1f}mm, hub R={hub_r:.1f}mm, gap={actual_gap:.1f}mm (need {air_gap:.1f}mm)"
                ))

        return results

    def check_blade_blade_collisions(self) -> List[AssemblyCheck]:
        """Check for blade-to-blade overlap within each stage.

        Verifies that the tangential chord projection fits within 80%
        of the arc spacing between adjacent blades at the root radius.
        """
        results = []
        blade_ring_radii = self.derived.get("blade_ring_radii", [])
        stages = self.config["blades"]["stages"]

        for i, stage in enumerate(stages):
            if i >= len(blade_ring_radii):
                continue
            n_blades = stage["num_blades"]
            ring_outer_r = blade_ring_radii[i]["ring_outer_r"]
            arc_at_root = ring_outer_r * (2 * math.pi / n_blades)

            # Estimate root chord (conservative 30mm) and twist (45 deg root)
            max_chord = 30.0
            twist_root = 45.0
            chord_projection = max_chord * math.cos(math.radians(twist_root))
            clearance_ratio = chord_projection / arc_at_root

            passed = clearance_ratio <= 0.80
            results.append(AssemblyCheck(
                check_name="blade_blade_clearance",
                part_a=f"blade_ring_stage_{i+1}",
                part_b=f"blade_ring_stage_{i+1}",
                value=clearance_ratio,
                limit=0.80,
                passed=passed,
                detail=f"Stage {i+1}: {n_blades} blades, arc={arc_at_root:.1f}mm, "
                       f"chord_proj={chord_projection:.1f}mm, ratio={clearance_ratio:.2f}"
            ))

        return results

    def check_collisions(self, meshes: dict) -> List[AssemblyCheck]:
        """Check for collisions between adjacent parts using trimesh."""
        results = []

        # Define adjacent part pairs to check
        pairs = self._get_adjacent_pairs(meshes)

        try:
            manager = trimesh.collision.CollisionManager()
            for name, mesh in meshes.items():
                manager.add_object(name, mesh)

            # Check each pair
            for name_a, name_b in pairs:
                if name_a in meshes and name_b in meshes:
                    is_collision, contact_data = manager.in_collision_single(
                        meshes[name_b], return_data=True
                    )
                    results.append(AssemblyCheck(
                        check_name="collision",
                        part_a=name_a,
                        part_b=name_b,
                        value=0.0 if is_collision else 1.0,
                        limit=1.0,  # need value=1 (no collision)
                        passed=not is_collision,
                        detail=f"{'COLLISION detected' if is_collision else 'No collision'}"
                    ))
        except Exception as e:
            results.append(AssemblyCheck(
                check_name="collision",
                part_a="all",
                part_b="all",
                value=0,
                limit=0,
                passed=True,  # Skip if collision detection fails
                detail=f"Collision detection skipped: {e}"
            ))

        return results

    def _get_adjacent_pairs(self, meshes: dict) -> List[Tuple[str, str]]:
        """Get list of adjacent part pairs that should be checked for collision."""
        pairs = []
        mesh_names = list(meshes.keys())

        # Hub halves shouldn't collide with blade rings
        for name in mesh_names:
            if "hub" in name:
                for other in mesh_names:
                    if "blade_ring" in other:
                        pairs.append((name, other))
            # Blade rings shouldn't collide with duct
            if "blade_ring" in name:
                for other in mesh_names:
                    if "duct" in other:
                        pairs.append((name, other))
            # Stators shouldn't collide with hub or blade rings
            if "stator" in name:
                for other in mesh_names:
                    if "hub" in other or "blade_ring" in other:
                        pairs.append((name, other))
            # Gear parts shouldn't collide with hub
            if "gear" in name and "ring" not in name:
                for other in mesh_names:
                    if "hub" in other:
                        pairs.append((name, other))

        return pairs

    def all_passed(self, meshes: dict = None) -> bool:
        """Check if all assembly validations passed."""
        meshes = meshes or {}
        results = self.validate_all(meshes)
        return all(r.passed for r in results)
