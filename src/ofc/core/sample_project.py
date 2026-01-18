"""Helper utilities for creating sample projects for manual testing.

This module provides utilities to create sample projects and set them up
for manual testing. Only runs when explicitly called.
"""

import shutil
from pathlib import Path
from typing import Optional

from .project import OceanProject


def create_sample_project(
    project_path: str | Path, *, name: Optional[str] = None
) -> OceanProject:
    """
    Create a sample project at the given path.

    Args:
        project_path: Path where project should be created
        name: Optional project name (defaults to directory name)

    Returns:
        OceanProject instance

    Raises:
        ValueError: If project already exists or path is invalid
    """
    return OceanProject.create(project_path, name=name)


def copy_images_to_project(
    project: OceanProject, source_paths: list[str | Path]
) -> list[str]:
    """
    Copy image files into project's raw images folder.

    Args:
        project: OceanProject instance
        source_paths: List of paths to image files to copy

    Returns:
        List of relative paths (forward slashes, relative to raw_images_folder) of copied images

    Raises:
        FileNotFoundError: If any source image does not exist
        ValueError: If any source path is not a valid image file
    """
    from .io_images import SUPPORTED_EXTENSIONS

    copied_paths = []

    for source_path in source_paths:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source image does not exist: {source}")

        if not source.is_file():
            raise ValueError(f"Source path is not a file: {source}")

        # Check extension
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Source file {source} does not have a supported image extension"
            )

        # Copy to project raw images folder
        dest = project.paths.raw_images_folder / source.name

        # Handle name conflicts
        counter = 1
        original_dest = dest
        while dest.exists():
            stem = original_dest.stem
            suffix = original_dest.suffix
            dest = project.paths.raw_images_folder / f"{stem}_{counter}{suffix}"
            counter += 1

        shutil.copy2(source, dest)

        # Get relative path (relative to raw_images_folder, not project root)
        rel_path = dest.relative_to(project.paths.raw_images_folder)
        rel_path_str = str(rel_path).replace("\\", "/")
        copied_paths.append(rel_path_str)

    return copied_paths


# Manual smoke test instructions:
#
# To manually test the Step 2 implementation:
#
# 1. Create a project:
#    >>> from ofc.core import OceanProject
#    >>> project = OceanProject.create("./test_project", name="Test Project")
#
# 2. Copy a couple images into raw images folder (or use copy_images_to_project helper):
#    >>> from ofc.core.sample_project import copy_images_to_project
#    >>> copy_images_to_project(project, ["path/to/image1.jpg", "path/to/image2.png"])
#
# 3. List raw images:
#    >>> images = project.list_raw_images()
#    >>> print(images)
#
# 4. Enumerate tiles for first image:
#    >>> from ofc.core import enumerate_tiles_for_image
#    >>> grid = project.get_grid()
#    >>> tiles = enumerate_tiles_for_image(project, images[0], grid)
#    >>> print(f"Found {len(tiles)} tiles")
#
# 5. Get a tile image and save it:
#    >>> from ofc.core import get_tile_image
#    >>> tile = tiles[0]
#    >>> tile_img = get_tile_image(project, tile)
#    >>> tile_img.save("preview.png")
#
# 6. Ensure labels exist for all tiles:
#    >>> from ofc.core import ensure_image_tiles
#    >>> rows_added = ensure_image_tiles(project, images[0], grid)
#    >>> print(f"Added {rows_added} label rows")
