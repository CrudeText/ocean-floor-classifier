"""Utilities for generating sample/test data."""

import csv
import random
from pathlib import Path

from .grid import GridSpec
from .labels import LabelsStore
from .project import OceanProject


def generate_test_labels(
    project: OceanProject,
    grid: GridSpec,
    *,
    num_images: int = 5,
    label_probability: float = 0.6,
    classes: list[str] | None = None,
) -> int:
    """
    Generate random test labels for a project.

    Creates labels for tiles across multiple images with random class assignments.
    Useful for testing the training system.

    Args:
        project: OceanProject instance
        grid: GridSpec for tiling
        num_images: Number of images to generate labels for (uses first N images)
        label_probability: Probability that a tile will be labeled (0.0 to 1.0)
        classes: List of class names to use. If None, uses default classes.

    Returns:
        Number of labels created

    Raises:
        ValueError: If project has no images or invalid parameters
    """
    if label_probability < 0.0 or label_probability > 1.0:
        raise ValueError(f"label_probability must be in [0, 1], got {label_probability}")

    # Get default classes if not provided
    if classes is None:
        classes = ["sand", "rock", "coral", "seagrass", "mud"]

    # Get images from project
    image_paths = project.list_raw_images()
    if not image_paths:
        raise ValueError(f"Project has no images in raw images folder: {project.paths.raw_images_folder}")

    # Limit to num_images
    image_paths = image_paths[:num_images]

    # Load labels store
    labels = project.get_labels()

    # Generate labels for each image
    total_labels = 0

    for image_rel_path in image_paths:
        # Get image size to determine tile count
        from .io_images import get_image_size

        try:
            image_path = project.get_raw_image_path(image_rel_path)
            width, height = get_image_size(image_path)

            # Enumerate tiles for this image
            from .tiles import enumerate_tiles_for_image

            tiles = enumerate_tiles_for_image(project, image_rel_path, grid)

            # Generate labels for some tiles
            for tile in tiles:
                # Randomly decide if this tile should be labeled
                if random.random() < label_probability:
                    # Randomly assign a class
                    label = random.choice(classes)
                    labels.set(image_rel_path, tile.tile_i, tile.tile_j, label)
                    total_labels += 1

        except (FileNotFoundError, ValueError):
            # Skip images that don't exist or can't be read
            continue

    # Save labels
    labels.save(project.paths.data_labels)

    return total_labels


def generate_labels_for_all_images(
    project: OceanProject,
    grid: GridSpec,
    *,
    label_probability: float = 0.5,
    classes: list[str] | None = None,
) -> int:
    """
    Generate random test labels for ALL images in the project.
    
    This will add labels to images that don't have any labels yet,
    and can also add more labels to images that already have some.

    Args:
        project: OceanProject instance
        grid: GridSpec for tiling
        label_probability: Probability that a tile will be labeled (0.0 to 1.0)
        classes: List of class names to use. If None, uses default classes.

    Returns:
        Number of new labels created
    """
    if label_probability < 0.0 or label_probability > 1.0:
        raise ValueError(f"label_probability must be in [0, 1], got {label_probability}")

    # Get default classes if not provided
    if classes is None:
        classes = ["sand", "rock", "coral", "seagrass", "mud"]

    # Get all images from project
    image_paths = project.list_raw_images()
    if not image_paths:
        raise ValueError(f"Project has no images in raw images folder: {project.paths.raw_images_folder}")

    # Load labels store
    labels = project.get_labels()
    
    # Get images that already have some labels
    images_with_labels = set()
    for img, _, _, label in labels.iter_rows():
        if label and label.strip():
            images_with_labels.add(img)

    # Generate labels for each image
    total_labels = 0

    for image_rel_path in image_paths:
        # Get image size to determine tile count
        from .io_images import get_image_size

        try:
            image_path = project.get_raw_image_path(image_rel_path)
            width, height = get_image_size(image_path)

            # Enumerate tiles for this image
            from .tiles import enumerate_tiles_for_image

            tiles = enumerate_tiles_for_image(project, image_rel_path, grid)

            # Generate labels for some tiles
            for tile in tiles:
                # Check if this tile already has a label
                existing_label = labels.get(image_rel_path, tile.tile_i, tile.tile_j)
                
                # Only label if it doesn't already have a label, or randomly add more
                if not existing_label or not existing_label.strip():
                    # Randomly decide if this tile should be labeled
                    if random.random() < label_probability:
                        # Randomly assign a class
                        label = random.choice(classes)
                        labels.set(image_rel_path, tile.tile_i, tile.tile_j, label)
                        total_labels += 1

        except (FileNotFoundError, ValueError):
            # Skip images that don't exist or can't be read
            continue

    # Save labels
    labels.save(project.paths.data_labels)

    return total_labels


def auto_generate_test_labels_if_needed(project: OceanProject, grid: GridSpec) -> bool:
    """
    Automatically generate test labels if the project has few or no labels.

    Also sets up default classes if classes.json is empty.

    Args:
        project: OceanProject instance
        grid: GridSpec for tiling

    Returns:
        True if labels were generated, False otherwise
    """
    labels = project.get_labels()
    counts = labels.counts()

    # Count labeled tiles (exclude empty labels)
    total_labeled = sum(count for label, count in counts.items() if label and label.strip())

    # If we have very few labels (< 10), generate test data
    if total_labeled < 10:
        try:
            # Check if we have images
            image_paths = project.list_raw_images()
            if not image_paths:
                return False

            # Set up default classes if classes.json is empty
            classes_path = project.paths.configs_dir / "classes.json"
            import json

            default_classes = ["sand", "rock", "coral", "seagrass", "mud"]
            if classes_path.exists():
                try:
                    existing_classes = json.loads(classes_path.read_text())
                    if not existing_classes or len(existing_classes) == 0:
                        classes_path.write_text(json.dumps(default_classes, indent=2) + "\n")
                except Exception:
                    # If classes.json is invalid, overwrite it
                    classes_path.write_text(json.dumps(default_classes, indent=2) + "\n")
            else:
                classes_path.write_text(json.dumps(default_classes, indent=2) + "\n")

            # Generate test labels for ALL images
            # Use all available images, 50% labeling probability
            num_new_labels = generate_labels_for_all_images(
                project,
                grid,
                label_probability=0.5,
                classes=default_classes,
            )
            print(f"Generated {num_new_labels} new labels for all {len(image_paths)} images")
            
            # After generating labels, update classes.json to match actual classes used
            # This ensures classes.json reflects what's actually in the labels
            labels = project.get_labels()
            actual_classes = set()
            for _, _, _, label in labels.iter_rows():
                if label and label.strip():
                    actual_classes.add(label.strip())
            
            if actual_classes:
                # Update classes.json with actual classes found in labels
                actual_classes_list = sorted(list(actual_classes))
                classes_path.write_text(json.dumps(actual_classes_list, indent=2) + "\n")
                print(f"Updated classes.json with {len(actual_classes_list)} classes: {actual_classes_list}")
            
            return True
        except Exception:
            # Silently fail if generation doesn't work
            return False

    return False
