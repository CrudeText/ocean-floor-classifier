"""Main window implementation."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ofc.core import OceanProject

from .tabs.label_tab import LabelTab
from .tabs.train_tab import TrainTab


class MainWindow(QWidget):
    """Main application window."""

    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.project: OceanProject | None = None
        self.init_ui()
        # Auto-open default project for testing
        self.auto_open_default_project()

    def init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle("Ocean Floor Classifier")
        self.setMinimumSize(1200, 800)
        # Show maximized (full screen windowed)
        self.showMaximized()

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top bar
        top_bar = self.create_top_bar()
        main_layout.addLayout(top_bar)

        # Central tabs
        self.tabs = QTabWidget()
        self.label_tab = LabelTab()
        self.train_tab = TrainTab()
        self.tabs.addTab(self.label_tab, "Label")
        self.tabs.addTab(self.train_tab, "Train")
        main_layout.addWidget(self.tabs)

        # Project path label
        self.project_path_label = QLabel("No project")
        main_layout.addWidget(self.project_path_label)

    def auto_open_default_project(self):
        """Auto-open the default project for testing."""
        default_project_path = Path(
            r"C:\Users\willi\OneDrive\Bureau\Active Projects\OceanFloorClassifier\Testing\Projects\v0.2.0\Test 1"
        )
        
        if default_project_path.exists():
            try:
                project = OceanProject.open(default_project_path)
                self.set_project(project)
            except Exception as e:
                # Silently fail if project can't be opened (for testing)
                pass

    def create_top_bar(self) -> QHBoxLayout:
        """Create top bar with project controls."""
        layout = QHBoxLayout()

        # Open Project button
        open_btn = QPushButton("Open Project...")
        open_btn.clicked.connect(self.open_project)
        layout.addWidget(open_btn)

        # New Project button
        new_btn = QPushButton("New Project...")
        new_btn.clicked.connect(self.new_project)
        layout.addWidget(new_btn)

        layout.addStretch()

        return layout

    def open_project(self):
        """Open an existing project."""
        folder = QFileDialog.getExistingDirectory(
            self, "Open Project", "", QFileDialog.Option.ShowDirsOnly
        )

        if not folder:
            return

        try:
            project = OceanProject.open(folder)
            self.set_project(project)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to open project:\n{str(e)}"
            )

    def new_project(self):
        """Create a new project."""
        # Show project creation dialog first
        from .project_dialog import ProjectCreationDialog

        dialog = ProjectCreationDialog(self)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)  # Show in front
        dialog.raise_()  # Bring to front
        dialog.activateWindow()  # Activate window
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Get folder from dialog
        project_folder = dialog.get_project_folder()
        if not project_folder:
            QMessageBox.warning(
                self, "Error", "Please select a project folder before creating the project."
            )
            return

        project_path = project_folder

        # Check if folder is empty
        if project_path.exists() and any(project_path.iterdir()):
            reply = QMessageBox.question(
                self,
                "Folder Not Empty",
                f"The folder {project_path} is not empty. Create project anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        try:
            # Create project with default settings first
            project = OceanProject.create(
                project_path, 
                name=dialog.get_project_name() or None,
                raw_images_folder=dialog.get_raw_images_folder()
            )

            # Update grid with custom settings
            from ofc.core import GridSpec

            tile_w = dialog.get_tile_width()
            tile_h = dialog.get_tile_height()
            stride_x = dialog.get_stride_x() or tile_w  # 0 means use tile width
            stride_y = dialog.get_stride_y() or tile_h  # 0 means use tile height

            grid = GridSpec(
                tile_w=tile_w,
                tile_h=tile_h,
                stride_x=stride_x,
                stride_y=stride_y,
                offset_x=0,
                offset_y=0,
                edge_policy="drop",
            )
            grid.save(project.paths.configs_dir / "grid.json")

            # Create empty classes.json (classes can be added later)
            import json

            classes_path = project.paths.configs_dir / "classes.json"
            classes_path.write_text(json.dumps([], indent=2) + "\n")

            # Reload project to get updated grid
            project = OceanProject.open(project_path)
            self.set_project(project)

            QMessageBox.information(
                self, "Success", f"Project created at:\n{project.root}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to create project:\n{str(e)}"
            )

    def set_project(self, project: OceanProject):
        """Set the current project and update UI."""
        self.project = project

        # Update project path label
        project_path_str = str(project.root)
        self.project_path_label.setText(f"Project: {project_path_str}")

        # Load grid and labels
        try:
            grid = project.get_grid()
            labels = project.get_labels()

            # Auto-generate test labels if needed (for testing)
            from ofc.core import auto_generate_test_labels_if_needed
            labels_generated = auto_generate_test_labels_if_needed(project, grid)
            if labels_generated:
                # Reload labels after generation
                labels = project.get_labels()
                # Also reload classes since they may have been updated
                try:
                    # Force reload by getting classes again
                    _ = project.get_classes()
                except Exception:
                    pass

            # Pass to label tab
            self.label_tab.set_project(project, grid, labels)
            self.train_tab.set_project(project, grid, labels)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Warning",
                f"Project loaded but some components failed:\n{str(e)}",
            )