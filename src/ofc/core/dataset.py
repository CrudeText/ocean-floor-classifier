"""Dataset management functionality for training."""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .grid import GridSpec
from .labels import LabelsStore
from .project import OceanProject
from .tiles import TileRef, get_tile_image


@dataclass
class DatasetStats:
    """Statistics about a dataset."""

    total_labeled_tiles: int
    num_classes: int
    class_counts: dict[str, int]
    class_distribution: dict[str, float]  # Percentage of each class
    images_with_labels: int
    average_tiles_per_image: float


class TrainingDataset(Dataset):
    """
    PyTorch Dataset for loading labeled tiles from an OceanProject.

    This dataset loads tiles that have non-empty labels, converts them to
    PyTorch tensors, and maps class names to integer indices.
    """

    def __init__(
        self,
        project: OceanProject,
        grid: GridSpec,
        labels: LabelsStore,
        classes: list[str],
        *,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        cache: Optional[object] = None,  # TileCache type, avoiding circular import
        exclude_classes: Optional[list[str]] = None,
    ):
        """
        Initialize training dataset.

        Args:
            project: OceanProject instance
            grid: GridSpec for tiling
            labels: LabelsStore with tile labels
            classes: List of class names (order determines class indices)
            transform: Optional callable to transform PIL images
            cache: Optional TileCache for image caching
            exclude_classes: Optional list of class names to exclude

        Raises:
            ValueError: If classes list is empty or contains duplicates
        """
        if not classes:
            raise ValueError("Classes list cannot be empty")
        if len(classes) != len(set(classes)):
            raise ValueError("Classes list contains duplicates")

        self.project = project
        self.grid = grid
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.cache = cache
        self.exclude_classes = set(exclude_classes) if exclude_classes else set()

        # Create class name to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

        # Build list of labeled tiles
        self.tile_refs: list[tuple[TileRef, int]] = []  # (TileRef, class_idx)

        self._build_tile_list()

    def _build_tile_list(self) -> None:
        """Build the list of labeled tiles from the labels store."""
        # Get all labeled rows (non-empty labels)
        for image_rel_path, tile_i, tile_j, label in self.labels.iter_rows():
            # Skip unlabeled tiles
            if not label or label.strip() == "":
                continue

            # Skip excluded classes
            if label in self.exclude_classes:
                continue

            # Check if label is in our classes list
            if label not in self.class_to_idx:
                # Skip labels that aren't in the classes list
                continue

            # Get class index
            class_idx = self.class_to_idx[label]

            # Create TileRef (we need to get tile coordinates from grid)
            # For efficiency, we'll enumerate tiles for this image
            try:
                from .tiles import enumerate_tiles_for_image

                tiles = enumerate_tiles_for_image(self.project, image_rel_path, self.grid)
                # Find the matching tile
                matching_tile = None
                for tile in tiles:
                    if tile.tile_i == tile_i and tile.tile_j == tile_j:
                        matching_tile = tile
                        break

                if matching_tile is None:
                    # Tile doesn't exist in grid (might be edge tile that was dropped)
                    continue

                self.tile_refs.append((matching_tile, class_idx))

            except (FileNotFoundError, ValueError):
                # Image doesn't exist or can't be read
                continue

    def __len__(self) -> int:
        """Return the number of labeled tiles."""
        return len(self.tile_refs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get a tile image and its class index.

        Args:
            idx: Index into the dataset

        Returns:
            Tuple of (image_tensor, class_index)
            - image_tensor: PyTorch tensor of shape (C, H, W) with values in [0, 1]
            - class_index: Integer class index

        Raises:
            IndexError: If idx is out of range
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        if idx < 0 or idx >= len(self.tile_refs):
            raise IndexError(f"Index {idx} out of range [0, {len(self.tile_refs)})")

        tile_ref, class_idx = self.tile_refs[idx]

        # Load tile image
        tile_img = get_tile_image(
            self.project, tile_ref, cache=self.cache, pad_value=(0, 0, 0)
        )

        # Apply transform if provided
        if self.transform is not None:
            tile_img = self.transform(tile_img)

        # Convert PIL Image to PyTorch tensor
        # PIL Image is (H, W, C) with values in [0, 255]
        # PyTorch expects (C, H, W) with values in [0, 1]
        import torchvision.transforms.functional as TF

        tensor = TF.to_tensor(tile_img)  # Converts to (C, H, W) and normalizes to [0, 1]

        return tensor, class_idx


def get_train_val_split(
    dataset: Dataset,
    val_split: float = 0.2,
    seed: Optional[int] = None,
) -> tuple[Dataset, Dataset]:
    """
    Split a dataset into training and validation sets.

    Args:
        dataset: Dataset to split
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        seed: Optional random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)

    Raises:
        ValueError: If val_split is not in [0, 1]
    """
    if not 0.0 <= val_split <= 1.0:
        raise ValueError(f"val_split must be in [0, 1], got {val_split}")

    if seed is not None:
        random.seed(seed)

    # Get dataset size
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Shuffle indices
    random.shuffle(indices)

    # Split indices
    val_size = int(dataset_size * val_split)
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])

    # Create subset datasets
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, sorted(train_indices))
    val_dataset = Subset(dataset, sorted(val_indices))

    return train_dataset, val_dataset


def get_class_weights(
    dataset: TrainingDataset,
    method: str = "balanced",
) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced datasets.

    Args:
        dataset: TrainingDataset instance
        method: Weight calculation method
            - "balanced": Inverse frequency weighting (sklearn style)
            - "inverse": Simple inverse frequency
            - "uniform": All weights equal to 1.0

    Returns:
        Tensor of shape (num_classes,) with weight for each class

    Raises:
        ValueError: If method is invalid
    """
    if method not in ("balanced", "inverse", "uniform"):
        raise ValueError(f"Invalid method: {method}. Must be 'balanced', 'inverse', or 'uniform'")

    if method == "uniform":
        num_classes = len(dataset.classes)
        return torch.ones(num_classes, dtype=torch.float32)

    # Count samples per class
    class_counts = Counter()
    for _, class_idx in dataset.tile_refs:
        class_counts[class_idx] += 1

    num_classes = len(dataset.classes)
    total_samples = sum(class_counts.values())

    if method == "balanced":
        # sklearn-style balanced weights: n_samples / (n_classes * count)
        weights = torch.zeros(num_classes, dtype=torch.float32)
        for class_idx, count in class_counts.items():
            if count > 0:
                weights[class_idx] = total_samples / (num_classes * count)
    else:  # inverse
        # Simple inverse frequency
        weights = torch.zeros(num_classes, dtype=torch.float32)
        for class_idx, count in class_counts.items():
            if count > 0:
                weights[class_idx] = total_samples / count

    # Normalize so minimum weight is 1.0 (optional, but common practice)
    min_weight = weights[weights > 0].min() if (weights > 0).any() else 1.0
    weights = weights / min_weight

    return weights


def get_dataset_stats(project: OceanProject, grid: GridSpec, labels: LabelsStore) -> DatasetStats:
    """
    Get statistics about the labeled dataset.

    Args:
        project: OceanProject instance
        grid: GridSpec for tiling
        labels: LabelsStore with tile labels

    Returns:
        DatasetStats object with dataset statistics
    """
    # Count labeled tiles and classes
    class_counts: dict[str, int] = defaultdict(int)
    images_with_labels: set[str] = set()
    total_labeled = 0

    for image_rel_path, tile_i, tile_j, label in labels.iter_rows():
        if label and label.strip() != "":
            class_counts[label] += 1
            images_with_labels.add(image_rel_path)
            total_labeled += 1

    num_classes = len(class_counts)
    total = total_labeled if total_labeled > 0 else 1  # Avoid division by zero

    # Calculate class distribution (percentages)
    class_distribution = {
        class_name: (count / total) * 100.0
        for class_name, count in class_counts.items()
    }

    # Calculate average tiles per image
    num_images = len(images_with_labels)
    avg_tiles_per_image = total_labeled / num_images if num_images > 0 else 0.0

    return DatasetStats(
        total_labeled_tiles=total_labeled,
        num_classes=num_classes,
        class_counts=dict(class_counts),
        class_distribution=class_distribution,
        images_with_labels=num_images,
        average_tiles_per_image=avg_tiles_per_image,
    )
