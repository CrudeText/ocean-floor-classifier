"""Project management functionality."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ProjectPaths:
    """Canonical paths for a project."""

    project_root: Path
    raw_images_folder: Path  # External folder path (not inside project)
    data_labels: Path
    configs_dir: Path
    runs_dir: Path
    runs_train: Path
    runs_infer: Path
    exports_dir: Path
    exports_tiles: Path

    @classmethod
    def from_root(cls, root: Path, raw_images_folder: Path | None = None) -> "ProjectPaths":
        """
        Create ProjectPaths from project root directory.
        
        Args:
            root: Project root directory
            raw_images_folder: Path to external raw images folder. If None, uses default.
        """
        root = Path(root).resolve()
        
        # Default raw images folder for testing
        if raw_images_folder is None:
            default_path = Path(
                r"D:\A - Project Data\OceanFloorClassifier\Usable Data\JPG"
            )
            raw_images_folder = default_path if default_path.exists() else root / "data" / "raw"
        else:
            raw_images_folder = Path(raw_images_folder).resolve()
        
        return cls(
            project_root=root,
            raw_images_folder=raw_images_folder,
            data_labels=root / "data" / "labels.csv",
            configs_dir=root / "configs",
            runs_dir=root / "runs",
            runs_train=root / "runs" / "train",
            runs_infer=root / "runs" / "infer",
            exports_dir=root / "exports",
            exports_tiles=root / "exports" / "tiles",
        )


class OceanProject:
    """Manages an ocean floor classifier project."""

    PROJECT_JSON = "project.json"
    VERSION = "0.2.0"

    def __init__(self, root: Path, name: Optional[str] = None, raw_images_folder: Path | None = None):
        """
        Initialize project with root path and optional name.
        
        Args:
            root: Project root directory
            name: Optional project name
            raw_images_folder: Path to external raw images folder
        """
        self.root = Path(root).resolve()
        self.paths = ProjectPaths.from_root(self.root, raw_images_folder)
        self.name = name or self.root.name
        self.created_utc: Optional[str] = None
        self.updated_utc: Optional[str] = None

    @classmethod
    def create(
        cls, path: str | Path, *, name: Optional[str] = None, raw_images_folder: Path | str | None = None
    ) -> "OceanProject":
        """
        Create a new project at the given path.

        Args:
            path: Root directory for the project
            name: Optional project name (defaults to directory name)
            raw_images_folder: Path to external raw images folder. If None, uses default.

        Returns:
            OceanProject instance

        Raises:
            ValueError: If project already exists or path is invalid
        """
        root = Path(path).resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Directory {root} already exists and is not empty")

        # Convert raw_images_folder to Path if string
        if raw_images_folder is not None:
            raw_images_folder = Path(raw_images_folder)

        # Create project instance
        project = cls(root, name=name, raw_images_folder=raw_images_folder)

        # Create directory structure (don't create raw_images_folder, it's external)
        project.paths.configs_dir.mkdir(parents=True, exist_ok=True)
        project.paths.runs_train.mkdir(parents=True, exist_ok=True)
        project.paths.runs_infer.mkdir(parents=True, exist_ok=True)
        project.paths.exports_tiles.mkdir(parents=True, exist_ok=True)

        # Create labels.csv with header only
        project.paths.data_labels.write_text("image_rel_path,tile_i,tile_j,label\n")

        # Create default grid.json
        from .grid import GridSpec

        default_grid = GridSpec(
            tile_w=256,
            tile_h=256,
            stride_x=256,
            stride_y=256,
            offset_x=0,
            offset_y=0,
            edge_policy="drop",
        )
        grid_path = project.paths.configs_dir / "grid.json"
        default_grid.save(grid_path)

        # Create default classes.json (empty list)
        classes_path = project.paths.configs_dir / "classes.json"
        classes_path.write_text(json.dumps([], indent=2) + "\n")

        # Set timestamps and save project.json
        now_iso = datetime.now(timezone.utc).isoformat()
        project.created_utc = now_iso
        project.updated_utc = now_iso
        project.save_project_json()

        return project

    @classmethod
    def open(cls, path: str | Path) -> "OceanProject":
        """
        Open an existing project.

        Args:
            path: Root directory of the project

        Returns:
            OceanProject instance

        Raises:
            ValueError: If project structure is invalid
        """
        root = Path(path).resolve()
        project = cls(root)
        project.load_project_json()
        project.validate()
        return project

    def validate(self) -> None:
        """
        Validate project structure.

        Raises:
            ValueError: If required directories or files are missing
        """
        errors = []

        if not self.paths.project_root.exists():
            errors.append(f"Project root does not exist: {self.paths.project_root}")

        # Note: raw_images_folder is external, so we don't require it to exist
        # But we can warn if it doesn't
        if not self.paths.raw_images_folder.exists():
            # Just a warning, not an error - user might add images later
            pass

        if not self.paths.data_labels.exists():
            errors.append(f"Labels CSV missing: {self.paths.data_labels}")

        if not self.paths.configs_dir.exists():
            errors.append(f"Configs directory missing: {self.paths.configs_dir}")

        grid_json = self.paths.configs_dir / "grid.json"
        if not grid_json.exists():
            errors.append(f"Grid config missing: {grid_json}")

        classes_json = self.paths.configs_dir / "classes.json"
        if not classes_json.exists():
            errors.append(f"Classes config missing: {classes_json}")

        if not self.paths.runs_dir.exists():
            errors.append(f"Runs directory missing: {self.paths.runs_dir}")

        if not self.paths.exports_dir.exists():
            errors.append(f"Exports directory missing: {self.paths.exports_dir}")

        project_json = self.root / self.PROJECT_JSON
        if not project_json.exists():
            errors.append(f"Project JSON missing: {project_json}")

        if errors:
            raise ValueError("Project validation failed:\n  " + "\n  ".join(errors))

    def save_project_json(self) -> None:
        """Save project metadata to project.json."""
        self.updated_utc = datetime.now(timezone.utc).isoformat()
        if self.created_utc is None:
            self.created_utc = self.updated_utc

        data = {
            "name": self.name,
            "version": self.VERSION,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "raw_images_folder": str(self.paths.raw_images_folder),
        }

        project_json = self.root / self.PROJECT_JSON
        project_json.write_text(json.dumps(data, indent=2) + "\n")

    def load_project_json(self) -> None:
        """Load project metadata from project.json."""
        project_json = self.root / self.PROJECT_JSON
        if not project_json.exists():
            raise ValueError(f"Project JSON not found: {project_json}")

        data = json.loads(project_json.read_text())
        self.name = data.get("name", self.root.name)
        self.created_utc = data.get("created_utc")
        self.updated_utc = data.get("updated_utc")
        
        # Load raw_images_folder if present, otherwise use default
        raw_folder_str = data.get("raw_images_folder")
        if raw_folder_str:
            self.paths.raw_images_folder = Path(raw_folder_str).resolve()
        else:
            # Use default for backward compatibility or set default
            default_path = Path(
                r"D:\A - Project Data\OceanFloorClassifier\Usable Data\JPG"
            )
            if default_path.exists():
                self.paths.raw_images_folder = default_path
            else:
                # Fallback to old location
                self.paths.raw_images_folder = self.root / "data" / "raw"

    @property
    def raw_dir(self) -> Path:
        """Get path to raw images directory."""
        return self.paths.raw_images_folder

    def list_raw_images(self) -> list[str]:
        """
        List all raw images in the raw images folder.

        Returns:
            List of image relative paths (forward slashes, relative to raw_images_folder)
        """
        from .io_images import list_images

        if not self.paths.raw_images_folder.exists():
            return []

        image_paths = list_images(self.paths.raw_images_folder)
        # Convert to relative paths with forward slashes (relative to raw_images_folder)
        rel_paths = []
        for img_path in image_paths:
            rel_path = img_path.relative_to(self.paths.raw_images_folder)
            # Use forward slashes for consistency (OS-agnostic)
            rel_path_str = str(rel_path).replace("\\", "/")
            rel_paths.append(rel_path_str)

        return rel_paths

    def get_raw_image_path(self, image_rel_path: str) -> Path:
        """
        Get full path to a raw image from relative path.

        Args:
            image_rel_path: Relative path from raw_images_folder (forward slashes OK)

        Returns:
            Full Path to image file

        Raises:
            ValueError: If path is outside raw_images_folder
        """
        # Normalize path separators
        normalized = image_rel_path.replace("\\", "/")
        # Remove leading slash if present
        if normalized.startswith("/"):
            normalized = normalized[1:]

        full_path = self.paths.raw_images_folder / normalized

        # Security check: ensure path is within raw_images_folder
        try:
            full_path.resolve().relative_to(self.paths.raw_images_folder.resolve())
        except ValueError:
            raise ValueError(
                f"Image path {image_rel_path} is outside raw images folder"
            ) from None

        return full_path
    
    def set_raw_images_folder(self, folder_path: Path | str) -> None:
        """
        Set the raw images folder path.
        
        Args:
            folder_path: Path to the raw images folder
        """
        self.paths.raw_images_folder = Path(folder_path).resolve()
        self.save_project_json()

    def get_grid(self) -> "GridSpec":
        """
        Load and return the grid specification.

        Returns:
            GridSpec instance

        Raises:
            FileNotFoundError: If grid.json does not exist
            ValueError: If grid.json is invalid
        """
        from .grid import GridSpec

        grid_path = self.paths.configs_dir / "grid.json"
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid config not found: {grid_path}")

        return GridSpec.load(grid_path)

    def get_labels(self) -> "LabelsStore":
        """
        Load and return the labels store.

        Returns:
            LabelsStore instance
        """
        from .labels import LabelsStore

        return LabelsStore.load(self.paths.data_labels)
    
    def get_classes(self) -> list[str]:
        """
        Load and return the list of classes.

        Returns:
            List of class names

        Raises:
            FileNotFoundError: If classes.json does not exist
            ValueError: If classes.json is invalid
        """
        classes_path = self.paths.configs_dir / "classes.json"
        if not classes_path.exists():
            raise FileNotFoundError(f"Classes config not found: {classes_path}")
        
        try:
            classes = json.loads(classes_path.read_text())
            if not isinstance(classes, list):
                raise ValueError(f"Classes must be a list, got {type(classes)}")
            # Ensure all items are strings
            return [str(c) for c in classes]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in classes.json: {e}") from e