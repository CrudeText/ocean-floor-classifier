"""Label management functionality."""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


class LabelsStore:
    """Manages label storage in CSV format."""

    CSV_HEADER = ["image_rel_path", "tile_i", "tile_j", "label"]

    def __init__(self):
        """Initialize empty labels store."""
        # Use dict keyed by (image_rel_path, i, j) to avoid duplicates
        self._data: dict[tuple[str, int, int], str] = {}

    @classmethod
    def load(cls, path: str | Path) -> "LabelsStore":
        """
        Load labels from CSV file.

        Args:
            path: Path to labels CSV file

        Returns:
            LabelsStore instance
        """
        store = cls()
        path = Path(path)

        if not path.exists():
            # Create empty file with header
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(cls.CSV_HEADER)
            return store

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Validate header
            if reader.fieldnames != cls.CSV_HEADER:
                raise ValueError(
                    f"Invalid CSV header. Expected {cls.CSV_HEADER}, "
                    f"got {reader.fieldnames}"
                )

            for row in reader:
                image_rel_path = row["image_rel_path"]
                tile_i = int(row["tile_i"])
                tile_j = int(row["tile_j"])
                label = row["label"]
                key = (image_rel_path, tile_i, tile_j)
                store._data[key] = label

        return store

    def save(self, path: str | Path) -> None:
        """
        Save labels to CSV file in deterministic order.

        Args:
            path: Path to save labels CSV file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sort by image_rel_path, then j, then i
        sorted_items = sorted(
            self._data.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])
        )

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADER)
            for (image_rel_path, tile_i, tile_j), label in sorted_items:
                writer.writerow([image_rel_path, tile_i, tile_j, label])

    def get(
        self, image_rel_path: str, i: int, j: int
    ) -> Optional[str]:
        """
        Get label for a specific tile.

        Args:
            image_rel_path: Relative path to image (from project root)
            i: Tile column index
            j: Tile row index

        Returns:
            Label string, or None if not found
        """
        key = (image_rel_path, i, j)
        return self._data.get(key)

    def set(self, image_rel_path: str, i: int, j: int, label: str) -> None:
        """
        Set label for a specific tile.

        Args:
            image_rel_path: Relative path to image (from project root)
            i: Tile column index
            j: Tile row index
            label: Label string (empty string for unlabeled)
        """
        key = (image_rel_path, i, j)
        self._data[key] = label

    def remove(self, image_rel_path: str, i: int, j: int) -> None:
        """
        Remove label for a specific tile.

        Args:
            image_rel_path: Relative path to image (from project root)
            i: Tile column index
            j: Tile row index
        """
        key = (image_rel_path, i, j)
        self._data.pop(key, None)

    def counts(self) -> dict[str, int]:
        """
        Get count of labels by label value.

        Returns:
            Dictionary mapping label -> count, including "" for unlabeled
        """
        counts: dict[str, int] = defaultdict(int)
        for label in self._data.values():
            counts[label] += 1
        return dict(counts)

    def iter_rows(self) -> Iterable[tuple[str, int, int, str]]:
        """
        Iterate over all label rows in deterministic order.

        Yields:
            Tuples of (image_rel_path, tile_i, tile_j, label)
        """
        sorted_items = sorted(
            self._data.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])
        )
        for (image_rel_path, tile_i, tile_j), label in sorted_items:
            yield (image_rel_path, tile_i, tile_j, label)

    def ensure_rows_for_image(
        self, image_rel_path: str, tile_indices: Iterable[tuple[int, int]]
    ) -> None:
        """
        Ensure rows exist for given tile indices, adding missing ones with empty label.

        Args:
            image_rel_path: Relative path to image (from project root)
            tile_indices: Iterable of (i, j) tile indices
        """
        for i, j in tile_indices:
            key = (image_rel_path, i, j)
            if key not in self._data:
                self._data[key] = ""


def ensure_image_tiles(
    project: "OceanProject", image_rel_path: str, grid: "GridSpec"
) -> int:
    """
    Ensure labels.csv has rows for all tiles of an image.

    Args:
        project: OceanProject instance
        image_rel_path: Relative path to image from project root
        grid: GridSpec for tiling

    Returns:
        Number of rows added

    Raises:
        FileNotFoundError: If image does not exist
        ValueError: If image cannot be read
    """
    from .tiles import enumerate_tiles_for_image

    # Enumerate tiles for the image
    tiles = enumerate_tiles_for_image(project, image_rel_path, grid)

    # Get tile indices
    tile_indices = [(tile.tile_i, tile.tile_j) for tile in tiles]

    # Load labels store
    labels = project.get_labels()

    # Count existing rows before
    existing_count = len(labels._data)

    # Ensure rows exist
    labels.ensure_rows_for_image(image_rel_path, tile_indices)

    # Count rows after
    new_count = len(labels._data)
    rows_added = new_count - existing_count

    # Save labels back
    labels.save(project.paths.data_labels)

    return rows_added
