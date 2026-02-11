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
    # Hub OD must fit motor + gears + bearings + walls
    motor_od = motor["body_diameter"]
    # Hub OD based on ring gear + wall thickness
    hub_od = ring_pd + 2 * module + 2 * hub_cfg["wall_thickness"]  # ring OD + walls
    hub_od = max(hub_od, motor_od + 2 * hub_cfg["wall_thickness"])
    derived["hub_od"] = hub_od

    # Hub length: motor + gear stages + coupling zones
    motor_len = motor["body_length"]
    gear_len = gears["gear_width"] * num_gear_stages
    coupling_zones = len(blades["stages"]) * 10  # ~10mm per coupling zone
    hub_length = motor_len + gear_len + coupling_zones + 20  # 20mm for bearings/spacing
    derived["hub_length"] = hub_length

    # --- Duct geometry ---
    duct_length = hub_length * duct["length_factor"]
    derived["duct_length"] = duct_length
    derived["duct_od"] = duct["inner_diameter"] + 2 * duct["wall_thickness"]

    # --- Blade geometry bounds ---
    blade_span = (duct["inner_diameter"] / 2 - blades["tip_clearance"]) - (hub_od / 2)
    derived["blade_span"] = blade_span
    derived["blade_hub_radius"] = hub_od / 2
    derived["blade_tip_radius"] = duct["inner_diameter"] / 2 - blades["tip_clearance"]

    # --- Angular velocity for each stage (rad/s) ---
    derived["stage_omega"] = [rpm * 2 * math.pi / 60 for rpm in stage_rpms]

    # --- Build volume check ---
    max_dim = max(
        derived["duct_od"],
        duct_length,
    )
    derived["max_part_dimension"] = max_dim
    derived["fits_build_volume"] = (
        derived["duct_od"] <= config["print"]["max_build_x"]
        and derived["duct_od"] <= config["print"]["max_build_y"]
        and duct_length <= config["print"]["max_build_z"]
    )

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
