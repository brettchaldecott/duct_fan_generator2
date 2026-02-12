"""Qt-based GUI viewer with config editor and 3D assembly view.

Uses PySide6 for the Qt framework and pyvistaqt for embedding PyVista
into a Qt widget. Provides side-by-side YAML config editing and
interactive 3D visualization with generate/refresh controls.
"""

import os
import sys
import glob
import traceback

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QPushButton, QToolBar, QStatusBar, QFileDialog,
    QSplitter, QTreeWidget, QTreeWidgetItem, QMessageBox, QLabel,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QAction

import pyvista as pv
from pyvistaqt import QtInteractor

# Color scheme (same as viewer.py)
PART_COLORS = {
    "hub": (0.6, 0.6, 0.6),
    "blade_ring_stage_1": (0.2, 0.4, 0.8),
    "blade_ring_stage_2": (0.8, 0.2, 0.2),
    "blade_ring_stage_3": (0.2, 0.7, 0.3),
    "stator": (0.75, 0.75, 0.8),
    "duct": (0.9, 0.9, 0.95),
    "gear_sun": (0.85, 0.65, 0.13),
    "gear_planet": (0.72, 0.53, 0.04),
    "gear_ring": (0.93, 0.79, 0.28),
}


def _classify_part(name: str) -> str:
    """Map a part name to a color key."""
    name_lower = name.lower()
    for key in PART_COLORS:
        if key in name_lower:
            return key
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
    return "hub"


class GenerateWorker(QThread):
    """Background thread for STL generation."""
    finished = Signal(str)  # success message or empty
    error = Signal(str)     # error message

    def __init__(self, config_text: str, config_path: str):
        super().__init__()
        self.config_text = config_text
        self.config_path = config_path

    def run(self):
        try:
            import yaml
            import tempfile
            from src.config import load_config
            from src.bemt import BEMTSolver
            from src.assembly import AssemblyGenerator

            # Write config to temp file and load
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(self.config_text)
                tmp_path = f.name

            try:
                config = load_config(tmp_path)
            finally:
                os.unlink(tmp_path)

            # BEMT analysis
            solver = BEMTSolver(config)
            bemt_results = solver.solve_all_stages()

            # Generate
            generator = AssemblyGenerator(config, bemt_results)
            result = generator.generate_and_export()

            n_files = len(result["exported_files"])
            mesh_pass = sum(1 for r in result["mesh_results"] if r.passed)
            mesh_total = len(result["mesh_results"])
            asm_pass = sum(1 for r in result["assembly_results"] if r.passed)
            asm_total = len(result["assembly_results"])

            msg = (f"Exported {n_files} STLs. "
                   f"Mesh: {mesh_pass}/{mesh_total}, Assembly: {asm_pass}/{asm_total}")
            self.finished.emit(msg)

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class DuctFanViewer(QMainWindow):
    """Main application window with config editor + 3D viewer."""

    def __init__(self, config_path: str = None, output_dir: str = None):
        super().__init__()
        self.setWindowTitle("Ducted Fan Generator 2")
        self.resize(1600, 900)

        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "default.yaml"
        )
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "output"
        )
        self.worker = None
        self.part_actors = {}

        self._setup_ui()
        self._load_config_file()
        self._load_stls()

    def _setup_ui(self):
        """Build the UI layout."""
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel: config editor + part tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("Configuration (YAML)"))
        self.config_editor = QTextEdit()
        self.config_editor.setFont(QFont("Courier New", 11))
        self.config_editor.setMinimumWidth(400)
        left_layout.addWidget(self.config_editor, stretch=3)

        left_layout.addWidget(QLabel("Parts"))
        self.part_tree = QTreeWidget()
        self.part_tree.setHeaderLabels(["Part", "Visible"])
        self.part_tree.itemChanged.connect(self._on_part_visibility_changed)
        left_layout.addWidget(self.part_tree, stretch=1)

        splitter.addWidget(left_panel)

        # Right panel: 3D viewer
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.add_axes()
        splitter.addWidget(self.plotter.interactor)

        splitter.setSizes([500, 1100])

        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        load_action = QAction("Load Config", self)
        load_action.triggered.connect(self._on_load_config)
        toolbar.addAction(load_action)

        save_action = QAction("Save Config", self)
        save_action.triggered.connect(self._on_save_config)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        generate_action = QAction("Generate STLs", self)
        generate_action.triggered.connect(self._on_generate)
        toolbar.addAction(generate_action)

        refresh_action = QAction("Refresh View", self)
        refresh_action.triggered.connect(self._on_refresh)
        toolbar.addAction(refresh_action)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _load_config_file(self):
        """Load config YAML into the editor."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config_editor.setPlainText(f.read())
            self.status_bar.showMessage(f"Loaded config: {self.config_path}")

    def _load_stls(self):
        """Load STL files from output directory into 3D view."""
        self.plotter.clear()
        self.part_tree.clear()
        self.part_actors = {}

        if not os.path.exists(self.output_dir):
            self.status_bar.showMessage("No output directory found")
            return

        stl_files = sorted(glob.glob(os.path.join(self.output_dir, "*.stl")))
        if not stl_files:
            self.status_bar.showMessage("No STL files in output/")
            return

        for stl_path in stl_files:
            name = os.path.splitext(os.path.basename(stl_path))[0]
            try:
                mesh = pv.read(stl_path)
                part_type = _classify_part(name)
                color = PART_COLORS.get(part_type, (0.5, 0.5, 0.5))
                opacity = 0.3 if "duct" in name.lower() else 1.0

                actor = self.plotter.add_mesh(
                    mesh, color=color, opacity=opacity,
                    label=name, smooth_shading=True,
                )
                self.part_actors[name] = actor

                # Add to part tree
                item = QTreeWidgetItem([name, ""])
                item.setCheckState(1, Qt.Checked)
                item.setData(0, Qt.UserRole, name)
                self.part_tree.addTopLevelItem(item)
            except Exception as e:
                print(f"Error loading {name}: {e}")

        self.plotter.reset_camera()
        self.status_bar.showMessage(f"Loaded {len(stl_files)} parts")

    def _on_part_visibility_changed(self, item, column):
        """Toggle part visibility when checkbox changes."""
        if column != 1:
            return
        name = item.data(0, Qt.UserRole)
        visible = item.checkState(1) == Qt.Checked
        if name in self.part_actors:
            self.part_actors[name].SetVisibility(visible)
            self.plotter.render()

    def _on_load_config(self):
        """Open file dialog to load a config file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "YAML files (*.yaml *.yml)")
        if path:
            self.config_path = path
            self._load_config_file()

    def _on_save_config(self):
        """Save current editor content to config file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", self.config_path, "YAML files (*.yaml *.yml)")
        if path:
            with open(path, "w") as f:
                f.write(self.config_editor.toPlainText())
            self.config_path = path
            self.status_bar.showMessage(f"Saved config: {path}")

    def _on_generate(self):
        """Run STL generation in background thread."""
        if self.worker and self.worker.isRunning():
            self.status_bar.showMessage("Generation already in progress...")
            return

        self.status_bar.showMessage("Generating STL files...")
        self.worker = GenerateWorker(
            self.config_editor.toPlainText(), self.config_path)
        self.worker.finished.connect(self._on_generate_done)
        self.worker.error.connect(self._on_generate_error)
        self.worker.start()

    def _on_generate_done(self, msg):
        """Handle successful generation."""
        self.status_bar.showMessage(msg)
        self._load_stls()

    def _on_generate_error(self, msg):
        """Handle generation error."""
        self.status_bar.showMessage("Generation failed!")
        QMessageBox.critical(self, "Generation Error", msg)

    def _on_refresh(self):
        """Reload STL files from disk."""
        self._load_stls()

    def closeEvent(self, event):
        """Clean up plotter on close."""
        self.plotter.close()
        super().closeEvent(event)


def launch_gui(config_path: str = None, output_dir: str = None):
    """Launch the Qt GUI application."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = DuctFanViewer(config_path, output_dir)
    window.show()
    app.exec()
