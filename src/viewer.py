"""Interactive 3D assembly viewer using PyVista.

Loads STL files from the output directory and displays them in an
interactive 3D window with color-coding by part type.
"""

import os
import glob
from typing import Optional

import pyvista as pv


# Color scheme by part type
PART_COLORS = {
    "hub": (0.6, 0.6, 0.6),          # gray
    "blade_ring_stage_1": (0.2, 0.4, 0.8),  # blue
    "blade_ring_stage_2": (0.8, 0.2, 0.2),  # red
    "blade_ring_stage_3": (0.2, 0.7, 0.3),  # green
    "stator": (0.75, 0.75, 0.8),     # silver
    "duct": (0.9, 0.9, 0.95),        # near-white
    "gear_sun": (0.85, 0.65, 0.13),  # gold
    "gear_planet": (0.72, 0.53, 0.04),  # darker gold
    "gear_ring": (0.93, 0.79, 0.28),    # light gold
    "carrier": (0.5, 0.8, 0.5),       # light green
    "inner_shaft": (0.4, 0.4, 0.7),   # steel blue
    "middle_tube": (0.5, 0.5, 0.8),   # lighter blue
    "outer_tube": (0.6, 0.6, 0.9),    # lightest blue
    "coupling_disc": (0.9, 0.5, 0.2),     # orange
    "ring_output_hub": (0.7, 0.3, 0.6),   # purple
}


def _classify_part(name: str) -> str:
    """Map an STL filename to a part type key for coloring."""
    name_lower = name.lower()
    # Check specific multi-word keys first (before generic substring matches)
    if "coupling_disc" in name_lower:
        return "coupling_disc"
    if "ring_output_hub" in name_lower:
        return "ring_output_hub"
    for key in PART_COLORS:
        if key in name_lower:
            return key
    # Fallback classifications
    if "hub" in name_lower:
        return "hub"
    if "blade" in name_lower:
        return "blade_ring_stage_1"
    if "stator" in name_lower:
        return "stator"
    if "duct" in name_lower:
        return "duct"
    if "gear" in name_lower:
        if "sun" in name_lower:
            return "gear_sun"
        if "planet" in name_lower:
            return "gear_planet"
        if "ring" in name_lower:
            return "gear_ring"
    return "hub"  # default gray


def view_assembly(stl_dir: str, window_size: Optional[tuple] = None):
    """Load all STL files from a directory and display in an interactive viewer.

    Args:
        stl_dir: Path to directory containing STL files
        window_size: Optional (width, height) tuple for the viewer window
    """
    stl_files = sorted(glob.glob(os.path.join(stl_dir, "*.stl")))

    if not stl_files:
        print(f"No STL files found in {stl_dir}")
        return

    plotter = pv.Plotter(window_size=window_size or (1400, 900))
    plotter.set_background("white")

    # Load all meshes upfront
    loaded_parts = []
    for stl_path in stl_files:
        name = os.path.splitext(os.path.basename(stl_path))[0]
        mesh = pv.read(stl_path)
        part_type = _classify_part(name)
        color = PART_COLORS.get(part_type, (0.5, 0.5, 0.5))
        opacity = 0.3 if "duct" in name.lower() else 1.0
        loaded_parts.append((name, mesh, color, opacity))

    legend_entries = []

    def _add_parts(clip=False):
        """Add all parts to the plotter, optionally clipped."""
        plotter.clear()
        legend_entries.clear()
        for name, mesh, color, opacity in loaded_parts:
            if mesh.n_points == 0:
                continue
            display_mesh = mesh
            if clip:
                # Clip at Y=0 plane, showing Y>0 half to reveal internals
                display_mesh = mesh.clip('-y', invert=False)
            plotter.add_mesh(
                display_mesh,
                color=color,
                opacity=opacity,
                label=name,
                smooth_shading=True,
            )
            legend_entries.append([name, color])
        plotter.add_legend(
            legend_entries,
            bcolor=(1, 1, 1),
            face="circle",
            size=(0.2, 0.3),
        )
        plotter.add_axes()

    # Initial display (no clip)
    clip_state = {"active": False}

    def toggle_clip():
        """Toggle clip plane to reveal internal structure."""
        clip_state["active"] = not clip_state["active"]
        _add_parts(clip=clip_state["active"])
        plotter.render()

    _add_parts(clip=False)
    plotter.add_key_event('c', toggle_clip)
    plotter.add_text("Press 'C' for cross-section", position='lower_left', font_size=10)

    plotter.camera.zoom(0.8)
    print(f"Displaying {len(stl_files)} parts. Press 'C' to toggle cross-section. Close window to exit.")
    plotter.show()
