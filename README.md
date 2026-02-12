# Ducted Fan Generator 2

Parametric multi-stage counter-rotating ducted fan STL generator for 3D printing.

Generates publication-quality STL files for a complete ducted fan assembly including blades, hub, duct, stators, and planetary gearbox — all from a single YAML configuration file.

## Architecture

```
config/default.yaml  -->  BEMT Analysis  -->  Validation  -->  CAD Generation  -->  STL Export
                          (aerodynamic)      (structural,       (CadQuery)          (trimesh)
                                              resonance,
                                              gear, mesh,
                                              assembly)
```

**Pipeline stages:**

1. **Configuration** — Load and validate YAML parameters
2. **BEMT Analysis** — Blade Element Momentum Theory solves for thrust, torque, and optimal blade geometry per stage
3. **Validation** — Physics validators check structural margins, resonance separation, gear stress, coupling torque, and system convergence
4. **CAD Generation** — CadQuery builds parametric 3D solids for all parts
5. **Mesh Validation** — Trimesh checks watertightness, volume, and assembly fit
6. **STL Export** — Individual STL files per part, ready for slicing

## Quick Start

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run analysis (BEMT + all validators)
python main.py --analyze

# Generate STL files (gated by validation)
python main.py --generate

# Generate and view in 3D
python main.py --generate --view

# Open Qt GUI with config editor + 3D viewer
python main.py --gui

# Force generation despite validation failures
python main.py --generate --force
```

## Configuration

All parameters are in `config/default.yaml`. Key sections:

| Section | Parameters |
|---------|-----------|
| `motor` | shaft diameter, body size, RPM, stall torque, voltage |
| `duct` | inner diameter, wall thickness, bellmouth radius |
| `gears` | module, tooth counts (sun/planet/ring), pressure angle, face width, helix angle |
| `blades` | per-stage blade count, airfoil designations, tip clearance, compression ratio |
| `hub` | wall thickness, bearing press-fit clearance |
| `stators` | strut count, thickness, chord |
| `bearings` | ID/OD/width for inner shaft, middle tube, outer tube, blade ring |
| `materials` | density, elastic modulus, tensile strength for blade and mechanical parts |
| `print` | build volume limits, nozzle/layer settings |

### Key Design Parameters

- **Motor**: 4kW BLDC outrunner, 8.0 N-m stall torque, 12000 RPM
- **Duct**: 200mm inner diameter
- **Gears**: 2-stage planetary, module 2.0, ratio 2.0 per stage, double helical (30 deg)
- **Blades**: 3 counter-rotating stages (7, 9, 13 blades at 12000/6000/3000 RPM)
- **Compression**: Progressive hub taper for 1.15:1 pressure ratio

## CLI Commands

| Command | Description |
|---------|------------|
| `--analyze` | Run BEMT analysis and all validation checks, output report |
| `--generate` | Generate STL files (requires validation pass or `--force`) |
| `--validate-only` | Validate existing STL files in `output/` |
| `--view` | Open PyVista 3D viewer for STL files |
| `--gui` | Open Qt-based GUI with config editor and 3D viewer |
| `--config FILE` | Use custom YAML config instead of default |
| `--part NAME` | Generate only a specific part |
| `--force` | Generate STLs even if validation fails |

## Project Structure

```
duct_fan_generator2/
  config/
    default.yaml          # All design parameters
  src/
    config.py             # Config loading, validation, derived values
    airfoil.py            # NACA 4-digit airfoil generation
    bemt.py               # Blade Element Momentum Theory solver
    blade_generator.py    # Blade ring CAD generation (lofted airfoils)
    hub_generator.py      # Hub housing (two halves with motor pocket)
    stator_generator.py   # Structural stator struts
    duct_generator.py     # Duct shell with bellmouth inlet
    gear_generator.py     # Planetary gearbox (double helical involute)
    magnetic_coupling.py  # Magnetic coupling analysis
    assembly.py           # Assembly coordinator + STL export
    viewer.py             # PyVista 3D viewer
    qt_viewer.py          # Qt GUI with config editor + 3D view
    utils.py              # Shared geometry helpers
  validation/
    structural_validator.py   # Blade/hub stress checks
    resonance_validator.py    # Vibration frequency separation
    gear_validator.py         # Gear stress, contact ratio, backlash
    coupling_validator.py     # Magnetic coupling torque margins
    system_simulator.py       # Full system equilibrium simulation
    mesh_validator.py         # STL watertightness, volume, dimensions
    assembly_validator.py     # Collision detection, clearances, build volume
    report_generator.py       # Validation report + BOM output
  tests/
    conftest.py               # Shared fixtures (session-scoped config, BEMT)
    test_*.py                 # Pytest test suite
  output/                     # Generated STL files and reports
  main.py                     # CLI entry point
```

## Generated Parts

| Part | Description |
|------|------------|
| `hub_half_a` | Motor-side hub housing (motor pocket, shaft bearing) |
| `hub_half_b` | Output-side hub housing (gear cavities, output bearings) |
| `blade_ring_stage_N` | Blade ring with lofted airfoils + magnet pockets |
| `stator_entry` | Inlet guide vanes (hub-to-duct struts) |
| `stator_exit` | Exit guide vanes |
| `duct_section_N` | Duct shell with bellmouth (split if needed for build volume) |
| `gear_sun_stage_N` | Sun gear with shaft bore |
| `gear_planet_J_stage_N` | Planet gears at carrier positions |
| `gear_ring_stage_N` | Ring gear (internal teeth) |

## Validation Checks

- **Structural**: Blade root bending stress, centrifugal stress, hub hoop stress
- **Resonance**: Natural frequency separation from blade-pass and gear-mesh excitations
- **Gears**: Lewis bending stress, contact ratio, undercutting, backlash range
- **Coupling**: Magnetic coupling torque safety factor per stage
- **System**: Motor-gearbox-blade equilibrium convergence, power budget
- **Mesh**: Watertightness, positive volume, build volume fit
- **Assembly**: Bearing seats, magnet pockets, tip clearance, collision detection
