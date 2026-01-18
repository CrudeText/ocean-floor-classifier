"""Tests for project functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from ofc.core import OceanProject


def test_project_create_structure(tmp_path: Path):
    """Test that project.create() creates the correct folder structure."""
    project_root = tmp_path / "test_project"
    project = OceanProject.create(project_root, name="Test Project")

    # Check directories exist
    assert project.paths.data_raw.exists()
    assert project.paths.configs_dir.exists()
    assert project.paths.runs_train.exists()
    assert project.paths.runs_infer.exists()
    assert project.paths.exports_tiles.exists()

    # Check files exist
    assert project.paths.data_labels.exists()
    assert (project.paths.configs_dir / "grid.json").exists()
    assert (project.paths.configs_dir / "classes.json").exists()
    assert (project.root / "project.json").exists()

    # Check labels.csv has header only
    labels_content = project.paths.data_labels.read_text()
    assert labels_content == "image_rel_path,tile_i,tile_j,label\n"

    # Check classes.json is empty list
    classes_content = json.loads(
        (project.paths.configs_dir / "classes.json").read_text()
    )
    assert classes_content == []

    # Check project.json structure
    project_json = json.loads((project.root / "project.json").read_text())
    assert project_json["name"] == "Test Project"
    assert project_json["version"] == "0.2.0"
    assert "created_utc" in project_json
    assert "updated_utc" in project_json
    assert project_json["created_utc"] == project_json["updated_utc"]


def test_project_create_default_name(tmp_path: Path):
    """Test that project name defaults to directory name if not provided."""
    project_root = tmp_path / "my_project"
    project = OceanProject.create(project_root)

    project_json = json.loads((project.root / "project.json").read_text())
    assert project_json["name"] == "my_project"


def test_project_create_fails_if_exists(tmp_path: Path):
    """Test that create() fails if directory exists and is not empty."""
    project_root = tmp_path / "existing_project"
    project_root.mkdir()
    (project_root / "some_file.txt").write_text("content")

    with pytest.raises(ValueError, match="already exists and is not empty"):
        OceanProject.create(project_root)


def test_project_open(tmp_path: Path):
    """Test opening an existing project."""
    project_root = tmp_path / "test_project"
    project1 = OceanProject.create(project_root, name="Test Project")

    # Open the project
    project2 = OceanProject.open(project_root)

    assert project2.name == "Test Project"
    assert project2.root == project1.root
    assert project2.created_utc == project1.created_utc
    assert project2.updated_utc == project1.updated_utc


def test_project_validate_success(tmp_path: Path):
    """Test validation of a valid project."""
    project_root = tmp_path / "test_project"
    project = OceanProject.create(project_root)

    # Should not raise
    project.validate()


def test_project_validate_missing_directory(tmp_path: Path):
    """Test validation fails when directory is missing."""
    project_root = tmp_path / "test_project"
    project = OceanProject.create(project_root)

    # Remove a required directory
    project.paths.data_raw.rmdir()

    with pytest.raises(ValueError, match="Data raw directory missing"):
        project.validate()


def test_project_validate_missing_file(tmp_path: Path):
    """Test validation fails when required file is missing."""
    project_root = tmp_path / "test_project"
    project = OceanProject.create(project_root)

    # Remove a required file
    project.paths.data_labels.unlink()

    with pytest.raises(ValueError, match="Labels CSV missing"):
        project.validate()


def test_project_save_and_load_json(tmp_path: Path):
    """Test saving and loading project JSON."""
    project_root = tmp_path / "test_project"
    project = OceanProject.create(project_root, name="Test Project")

    # Modify and save
    project.name = "Updated Name"
    project.save_project_json()

    # Load in new instance
    project2 = OceanProject(project_root)
    project2.load_project_json()

    assert project2.name == "Updated Name"
    assert project2.created_utc == project.created_utc
    assert project2.updated_utc is not None
    assert project2.updated_utc != project.created_utc  # Should be updated


def test_project_paths_from_root():
    """Test ProjectPaths.from_root() creates correct paths."""
    from ofc.core import ProjectPaths

    root = Path("/some/project/root")
    paths = ProjectPaths.from_root(root)

    assert paths.project_root == root.resolve()
    assert paths.data_raw == root.resolve() / "data" / "raw"
    assert paths.data_labels == root.resolve() / "data" / "labels.csv"
    assert paths.configs_dir == root.resolve() / "configs"
    assert paths.runs_dir == root.resolve() / "runs"
    assert paths.runs_train == root.resolve() / "runs" / "train"
    assert paths.runs_infer == root.resolve() / "runs" / "infer"
    assert paths.exports_dir == root.resolve() / "exports"
    assert paths.exports_tiles == root.resolve() / "exports" / "tiles"