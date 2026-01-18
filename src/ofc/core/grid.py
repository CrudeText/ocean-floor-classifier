"""Grid functionality for tile management."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


@dataclass
class GridSpec:
    """Specification for tiling an image into a grid."""

    tile_w: int
    tile_h: int
    stride_x: int
    stride_y: int
    offset_x: int = 0
    offset_y: int = 0
    edge_policy: Literal["drop", "pad"] = "drop"

    def to_dict(self) -> dict:
        """Convert GridSpec to dictionary."""
        return {
            "tile_w": self.tile_w,
            "tile_h": self.tile_h,
            "stride_x": self.stride_x,
            "stride_y": self.stride_y,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "edge_policy": self.edge_policy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridSpec":
        """Create GridSpec from dictionary."""
        return cls(
            tile_w=data["tile_w"],
            tile_h=data["tile_h"],
            stride_x=data["stride_x"],
            stride_y=data["stride_y"],
            offset_x=data.get("offset_x", 0),
            offset_y=data.get("offset_y", 0),
            edge_policy=data.get("edge_policy", "drop"),
        )

    def save(self, path: str | Path) -> None:
        """Save GridSpec to JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "GridSpec":
        """Load GridSpec from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def tile_rect(self, i: int, j: int) -> tuple[int, int, int, int]:
        """
        Get rectangle coordinates for tile at grid position (i, j).

        Args:
            i: Tile column index
            j: Tile row index

        Returns:
            Tuple of (x, y, w, h) where (x, y) is top-left corner
            and (w, h) are width and height
        """
        x = self.offset_x + i * self.stride_x
        y = self.offset_y + j * self.stride_y
        return (x, y, self.tile_w, self.tile_h)

    def iter_tiles_for_image(
        self, width: int, height: int
    ) -> Iterable[tuple[int, int, int, int, int, int]]:
        """
        Iterate over tiles for an image of given dimensions.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Yields:
            Tuples of (i, j, x, y, w, h) where:
            - i, j: Tile grid indices
            - x, y: Top-left corner coordinates
            - w, h: Tile width and height
        """
        # Calculate how many tiles fit
        max_i = (width - self.offset_x + self.stride_x - 1) // self.stride_x
        max_j = (height - self.offset_y + self.stride_y - 1) // self.stride_y

        for j in range(max_j):
            for i in range(max_i):
                x, y, w, h = self.tile_rect(i, j)

                if self.edge_policy == "drop":
                    # Only yield tiles that are completely inside image bounds
                    if x + w <= width and y + h <= height:
                        yield (i, j, x, y, w, h)
                elif self.edge_policy == "pad":
                    # Include all tiles, even if they extend beyond bounds
                    # Caller is responsible for padding when cropping
                    yield (i, j, x, y, w, h)
                else:
                    raise ValueError(f"Unknown edge_policy: {self.edge_policy}")