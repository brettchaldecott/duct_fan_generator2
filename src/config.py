"""Configuration loading, validation, and derived value computation."""

import os
import math
from copy import deepcopy
from typing import Any

import yaml


# Default config path
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "default.yaml"
)


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


def load_config(path: str = None) -> dict:
    """Load configuration from YAML file, validate, and compute derived values."""
    if path is None:
        path = DEFAULT_CONFIG_PATH

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)
    compute_derived(config)
    return config


def validate_config(config: dict) -> None:
    """Validate configuration constraints. Raises ConfigError on failure."""
    errors = []

    # --- Gear tooth constraint ---
    gears = config.get("gears", {})
    sun = gears.get("sun_teeth", 0)
    planet = gears.get("planet_teeth", 0)
    ring = gears.get("ring_teeth", 0)
    if ring != sun + 2 * planet:
        errors.append(
            f"Gear tooth constraint violated: ring_teeth ({ring}) "
            f"must equal sun_teeth + 2*planet_teeth ({sun + 2 * planet})"
        )

    # --- Positive dimensions ---
    duct_id = config.get("duct", {}).get("inner_diameter", 0)
    if duct_id <= 0:
        errors.append(f"Duct inner_diameter must be positive, got {duct_id}")

    hub_wall = config.get("hub", {}).get("wall_thickness", 0)
    if hub_wall <= 0:
        errors.append(f"Hub wall_thickness must be positive, got {hub_wall}")

    # --- Gear module and teeth must be positive ---
    if gears.get("module", 0) <= 0:
        errors.append("Gear module must be positive")
    if sun <= 0 or planet <= 0 or ring <= 0:
        errors.append("All gear tooth counts must be positive")

    # --- Bearing size consistency: inner < middle < outer ---
    bearings = config.get("bearings", {})
    inner_od = bearings.get("inner_shaft", {}).get("od", 0)
    middle_od = bearings.get("middle_tube", {}).get("od", 0)
    outer_od = bearings.get("outer_tube", {}).get("od", 0)
    if inner_od >= middle_od:
        errors.append(
            f"Inner shaft bearing OD ({inner_od}) must be less than "
            f"middle tube bearing OD ({middle_od})"
        )
    if middle_od >= outer_od:
        errors.append(
            f"Middle tube bearing OD ({middle_od}) must be less than "
            f"outer tube bearing OD ({outer_od})"
        )

    # --- Motor fits inside hub ---
    motor_diam = config.get("motor", {}).get("body_diameter", 0)
    # Hub OD must accommodate motor + wall thickness
    # (checked after derived values, but basic sanity here)
    if motor_diam <= 0:
        errors.append(f"Motor body_diameter must be positive, got {motor_diam}")

    # --- Build volume checks ---
    print_cfg = config.get("print", {})
    max_x = print_cfg.get("max_build_x", 300)
    max_y = print_cfg.get("max_build_y", 300)
    if duct_id > 0:
        duct_od = duct_id + 2 * config.get("duct", {}).get("wall_thickness", 4)
        if duct_od > max_x or duct_od > max_y:
            errors.append(
                f"Duct OD ({duct_od}mm) exceeds build volume "
                f"({max_x}x{max_y}mm)"
            )

    # --- Blade tip clearance ---
    tip_clearance = config.get("blades", {}).get("tip_clearance", 0)
    if tip_clearance <= 0:
        errors.append(f"Blade tip_clearance must be positive, got {tip_clearance}")

    # --- Magnetic coupling stages match blade stages ---
    num_blade_stages = len(config.get("blades", {}).get("stages", []))
    num_coupling_stages = len(
        config.get("magnetic_coupling", {}).get("stages", [])
    )
    if num_blade_stages != num_coupling_stages:
        errors.append(
            f"Number of blade stages ({num_blade_stages}) must match "
            f"coupling stages ({num_coupling_stages})"
        )

    # --- Motor RPM and stall torque ---
    motor = config.get("motor", {})
    if motor.get("rpm", 0) <= 0:
        errors.append("Motor RPM must be positive")
    if motor.get("stall_torque", 0) <= 0:
        errors.append("Motor stall_torque must be positive")
    if motor.get("no_load_rpm", 0) <= motor.get("rpm", 0):
        pass  # no_load_rpm should be > operating rpm, but not strictly required

    # --- Material properties ---
    for mat_key in ["blade", "mechanical"]:
        mat = config.get("materials", {}).get(mat_key, {})
        if mat.get("tensile_strength", 0) <= 0:
            errors.append(f"Material '{mat_key}' tensile_strength must be positive")
        if mat.get("density", 0) <= 0:
            errors.append(f"Material '{mat_key}' density must be positive")

    if errors:
        raise ConfigError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def compute_derived(config: dict) -> None:
    """Compute derived values and add them to the config dict."""
    derived = {}

    gears = config["gears"]
    motor = config["motor"]
    duct = config["duct"]
    blades = config["blades"]
    bearings = config["bearings"]
    hub_cfg = config["hub"]

    # --- Gear pitch diameters ---
    module = gears["module"]
    sun_teeth = gears["sun_teeth"]
    planet_teeth = gears["planet_teeth"]
    ring_teeth = gears["ring_teeth"]

    sun_pd = module * sun_teeth  # mm
    planet_pd = module * planet_teeth
    ring_pd = module * ring_teeth

    derived["sun_pitch_diameter"] = sun_pd
    derived["planet_pitch_diameter"] = planet_pd
    derived["ring_pitch_diameter"] = ring_pd

    # --- Gear ratio (carrier-fixed planetary) ---
    # With carrier fixed: ratio = -R/S (negative = reversal)
    gear_ratio = ring_teeth / sun_teeth  # magnitude
    derived["gear_ratio"] = gear_ratio  # 2.0 for default
    derived["gear_ratio_with_reversal"] = -gear_ratio  # negative = direction reversal

    # --- Stage RPMs and rotation directions ---
    # Stage 1: driven directly by motor shaft (or first gear stage output)
    # With 2 planetary stages: motor -> stage1 gear -> stage2 gear -> stage3
    # Each carrier-fixed stage: output_rpm = input_rpm / ratio, direction reverses
    motor_rpm = motor["rpm"]
    num_gear_stages = gears["num_stages"]

    stage_rpms = []
    stage_directions = []  # +1 = CW, -1 = CCW

    # Stage 1: directly from motor (through coupling)
    stage_rpms.append(float(motor_rpm))
    stage_directions.append(1)  # CW

    # Each gear stage reduces RPM and reverses direction
    current_rpm = float(motor_rpm)
    current_dir = 1
    for i in range(num_gear_stages):
        current_rpm = current_rpm / gear_ratio
        current_dir = -current_dir  # reversal
        stage_rpms.append(current_rpm)
        stage_directions.append(current_dir)

    derived["stage_rpms"] = stage_rpms
    derived["stage_directions"] = stage_directions

    # --- Hub geometry ---
    # Hub OD must fit ring gear outer wall + hub walls
    motor_od = motor["body_diameter"]
    # Ring gear outer wall radius = (root_diameter/2) + 2*module
    ring_root_d = ring_pd + 2 * 1.25 * module  # internal gear root is larger
    ring_outer_wall_d = ring_root_d + 4 * module  # wall beyond root
    hub_od = ring_outer_wall_d + 2 * hub_cfg["wall_thickness"]
    hub_od = max(hub_od, motor_od + 2 * hub_cfg["wall_thickness"])
    derived["hub_od"] = hub_od

    # Hub length and duct length are computed after axial layout (below).
    # Placeholder — will be set by layout computation.
    hub_length = 0  # computed later from layout

    # --- Blade geometry bounds ---
    # blade_hub_radius is the physical hub OD/2 (used for hub geometry)
    # blade_root_radius will be updated after blade ring radii are computed (below)
    derived["blade_hub_radius"] = hub_od / 2
    derived["blade_tip_radius"] = duct["inner_diameter"] / 2 - blades["tip_clearance"]
    blade_span = derived["blade_tip_radius"] - derived["blade_hub_radius"]
    derived["blade_span"] = blade_span

    # --- Per-stage hub radii for compression ---
    compression_ratio = blades.get("compression_ratio", 1.0)
    num_blade_stages = len(blades["stages"])
    tip_r = duct["inner_diameter"] / 2 - blades["tip_clearance"]

    if compression_ratio > 1.0 and num_blade_stages > 0:
        gamma = 1.4  # air
        per_stage_hub_radii = []
        area_current = math.pi * (tip_r**2 - (hub_od / 2)**2)
        pr_per_stage = compression_ratio ** (1.0 / num_blade_stages)

        for i in range(num_blade_stages):
            hub_r_i = math.sqrt(max(tip_r**2 - area_current / math.pi, (hub_od / 2)**2))
            per_stage_hub_radii.append(hub_r_i)
            # Reduce annulus area for next stage (isentropic compression)
            area_current /= pr_per_stage ** (1.0 / gamma)

        derived["per_stage_hub_radii"] = per_stage_hub_radii
    else:
        derived["per_stage_hub_radii"] = [hub_od / 2] * num_blade_stages

    # --- External blade ring geometry ---
    # Blade rings wrap OUTSIDE the hub housing with a radial air gap
    # for magnetic coupling. Magnets sit at the hub wall / ring inner wall interface.
    blade_ring_air_gap = 1.0  # mm radial gap for magnetic coupling
    blade_ring_wall_thickness = hub_cfg["wall_thickness"]  # mm structural wall
    blade_ring_radii = []
    for i in range(num_blade_stages):
        stage_hub_r = derived["per_stage_hub_radii"][i]
        ring_inner_r = stage_hub_r + blade_ring_air_gap
        ring_outer_r = ring_inner_r + blade_ring_wall_thickness
        blade_ring_radii.append({
            "ring_inner_r": ring_inner_r,
            "ring_outer_r": ring_outer_r,
        })
    derived["blade_ring_radii"] = blade_ring_radii
    derived["blade_ring_air_gap"] = blade_ring_air_gap
    derived["blade_ring_wall_thickness"] = blade_ring_wall_thickness

    # Derive coupling_radius per stage (at hub outer wall / blade ring interface)
    coupling_stages = config["magnetic_coupling"]["stages"]
    for i in range(min(num_blade_stages, len(coupling_stages))):
        coupling_stages[i]["coupling_radius"] = derived["per_stage_hub_radii"][i]

    # Update blade_hub_radius and blade_span to reflect external ring architecture
    # Blades root at ring_outer_r, not at hub_od/2
    if blade_ring_radii:
        # Use the first stage ring outer as the reference blade root radius
        max_ring_outer_r = max(br["ring_outer_r"] for br in blade_ring_radii)
        derived["blade_hub_radius"] = max_ring_outer_r
        derived["blade_span"] = derived["blade_tip_radius"] - max_ring_outer_r

    # --- Angular velocity for each stage (rad/s) ---
    derived["stage_omega"] = [rpm * 2 * math.pi / 60 for rpm in stage_rpms]

    # --- Axial layout (Z positions in mm) ---
    # Layout: bellmouth | stator_entry | blade_1 | gear_0 | blade_2 | gear_1 | blade_3 | stator_exit
    # Gear stages are placed between their corresponding blade stages.
    stator_chord = config["stators"]["strut_chord"]

    # blade_axial_width: must accommodate the full axial projection of
    # the root airfoil (chord × sin(twist)) plus margin
    max_root_chord = 30.0   # mm (conservative estimate for root chord)
    max_twist_deg = 50.0    # degrees (conservative max root twist)
    max_axial_extent = max_root_chord * math.sin(math.radians(max_twist_deg)) + 2  # margin
    blade_axial_width = max(max_axial_extent, bearings["blade_ring"]["width"] + 2)
    # Result: ~25mm instead of old 10mm

    gear_width = gears["gear_width"]
    carrier_width = 3.0  # carrier plate thickness

    # inter_stage_gap: must clear blade overhang beyond the ring + stator chord
    blade_overhang = blade_axial_width / 2  # blade extends this far beyond ring center
    inter_stage_gap = max(blade_overhang + 3.0, 8.0)  # at least 3mm clearance beyond blade

    num_blade_stages = len(blades["stages"])

    positions = {}
    z = 0.0
    positions["stator_entry"] = z
    z += stator_chord + inter_stage_gap

    for i in range(num_blade_stages):
        # Position stores the CENTER of the blade ring (ring is centered at Z=0 locally)
        positions[f"blade_ring_stage_{i+1}"] = z + blade_axial_width / 2
        z += blade_axial_width

        # Place gear stage between blade stages (gear i goes after blade i+1)
        if i < num_gear_stages:
            z += inter_stage_gap
            positions[f"carrier_front_{i}"] = z
            z += carrier_width
            positions[f"gear_stage_{i}"] = z
            z += gear_width
            positions[f"carrier_back_{i}"] = z
            z += carrier_width

        if i < num_blade_stages - 1:
            z += inter_stage_gap

    z += inter_stage_gap
    positions["stator_exit"] = z
    z += stator_chord

    # Hub spans from first blade stage to last, with margins
    blade_1_center = positions["blade_ring_stage_1"]
    blade_start = blade_1_center - blade_axial_width / 2
    blade_n_center = positions[f"blade_ring_stage_{num_blade_stages}"]
    blade_end = blade_n_center + blade_axial_width / 2
    hub_margin = hub_cfg["wall_thickness"] + 5  # wall + bearing space
    # Extend hub to overlap with stator zones for structural continuity
    stator_entry_end = positions["stator_entry"] + stator_chord
    stator_exit_start = positions["stator_exit"]
    hub_start = min(blade_start - hub_margin, stator_entry_end - 2.0)
    hub_end = max(blade_end + hub_margin, stator_exit_start + 2.0)
    hub_length = hub_end - hub_start
    derived["hub_length"] = hub_length

    hub_center_z = (hub_start + hub_end) / 2
    positions["hub_half_a"] = hub_start
    positions["hub_half_b"] = hub_center_z

    # Duct covers the full assembly length plus bellmouth
    bellmouth_r = duct.get("bellmouth_radius", 15.0)
    positions["duct_section_1"] = -bellmouth_r

    # --- Duct geometry (derived from layout) ---
    duct_length = z + bellmouth_r + 5  # total assembly length + bellmouth + margin
    duct_length = max(duct_length, hub_length * duct["length_factor"])
    derived["duct_length"] = duct_length
    derived["duct_od"] = duct["inner_diameter"] + 2 * duct["wall_thickness"]

    # --- Build volume check ---
    max_dim = max(derived["duct_od"], duct_length)
    derived["max_part_dimension"] = max_dim
    derived["fits_build_volume"] = (
        derived["duct_od"] <= config["print"]["max_build_x"]
        and derived["duct_od"] <= config["print"]["max_build_y"]
        and duct_length <= config["print"]["max_build_z"]
    )

    derived["part_positions"] = positions
    derived["total_axial_length"] = z
    derived["blade_axial_width"] = blade_axial_width

    # --- Shaft/tube dimensions for concentric architecture ---
    derived["inner_shaft_diameter"] = motor["shaft_diameter"]  # 5mm
    derived["middle_tube_od"] = bearings["middle_tube"]["id"]  # 15mm
    derived["middle_tube_id"] = motor["shaft_diameter"] + 3.0  # clearance over shaft
    derived["outer_tube_od"] = bearings["outer_tube"]["id"]  # 25mm
    derived["outer_tube_id"] = derived["middle_tube_od"] + 2.0  # clearance over middle tube
    derived["ring_outer_wall_radius"] = ring_outer_wall_d / 2

    config["derived"] = derived


def get_material(config: dict, part_type: str) -> dict:
    """Get material properties for a part type.

    Args:
        config: Full configuration dict
        part_type: 'blade' or 'mechanical'

    Returns:
        Material properties dict
    """
    return config["materials"][part_type]


def get_stage_config(config: dict, stage_index: int) -> dict:
    """Get configuration for a specific blade/coupling stage.

    Args:
        config: Full configuration dict
        stage_index: 0-based stage index

    Returns:
        Dict with blade config, coupling config, RPM, direction, omega
    """
    derived = config["derived"]
    return {
        "blade": config["blades"]["stages"][stage_index],
        "coupling": config["magnetic_coupling"]["stages"][stage_index],
        "rpm": derived["stage_rpms"][stage_index],
        "direction": derived["stage_directions"][stage_index],
        "omega": derived["stage_omega"][stage_index],
        "hub_radius": derived["blade_hub_radius"],
        "tip_radius": derived["blade_tip_radius"],
    }
