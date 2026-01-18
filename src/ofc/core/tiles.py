"""Virtual tiling API for images."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from .grid import GridSpec
from .io_images import TileCache, get_image_size
from .project import OceanProject


@dataclass
class TileRef:
    """Reference to a tile within an image."""

    image_rel_path: str
    tile_i: int
    tile_j: int
    x: int
    y: int
    w: int
    h: int


def enumerate_tiles_for_image(
    project: OceanProject, image_rel_path: str, grid: GridSpec
) -> list[TileRef]:
    """
    Enumerate all tiles for an image according to grid specification.

    Args:
        project: OceanProject instance
        image_rel_path: Relative path to image from project root
        grid: GridSpec for tiling

    Returns:
        List of TileRef objects

    Raises:
        FileNotFoundError: If image does not exist
        ValueError: If image cannot be read
    """
    # Get full path to image
    image_path = project.get_raw_image_path(image_rel_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Get image dimensions
    width, height = get_image_size(image_path)

    # Enumerate tiles using grid
    tiles = []
    for i, j, x, y, w, h in grid.iter_tiles_for_image(width, height):
        tiles.append(
            TileRef(
                image_rel_path=image_rel_path,
                tile_i=i,
                tile_j=j,
                x=x,
                y=y,
                w=w,
                h=h,
            )
        )

    return tiles


def get_tile_image(
    project: OceanProject,
    tile: TileRef,
    *,
    cache: Optional[TileCache] = None,
    pad_value: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Get a cropped tile image from the source image.

    Args:
        project: OceanProject instance
        tile: TileRef specifying the tile
        cache: Optional TileCache for image caching
        pad_value: RGB tuple for padding (default: black)

    Returns:
        PIL Image of size (tile.w, tile.h) in RGB mode

    Raises:
        FileNotFoundError: If image does not exist
        ValueError: If image cannot be read or cropped
    """
    # Get full path to image
    image_path = project.get_raw_image_path(tile.image_rel_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image (from cache if available)
    if cache is not None:
        base_img = cache.get(image_path)
    else:
        from .io_images import read_image_pil

        base_img = read_image_pil(image_path)

    img_width, img_height = base_img.size

    # Calculate crop bounds
    x = tile.x
    y = tile.y
    w = tile.w
    h = tile.h

    # Check if tile extends beyond image bounds
    x_end = x + w
    y_end = y + h
    exceeds_bounds = x < 0 or y < 0 or x_end > img_width or y_end > img_height

    if exceeds_bounds:
        # Need to handle padding
        # Create a new image with the tile size, filled with pad_value
        tile_img = Image.new("RGB", (w, h), pad_value)

        # Calculate the actual crop region within image bounds
        crop_x = max(0, x)
        crop_y = max(0, y)
        crop_x_end = min(img_width, x_end)
        crop_y_end = min(img_height, y_end)

        # Calculate where to paste in the tile image
        paste_x = crop_x - x
        paste_y = crop_y - y

        # Crop the valid region from base image
        if crop_x < crop_x_end and crop_y < crop_y_end:
            crop_w = crop_x_end - crop_x
            crop_h = crop_y_end - crop_y
            cropped = base_img.crop((crop_x, crop_y, crop_x_end, crop_y_end))
            tile_img.paste(cropped, (paste_x, paste_y))

        return tile_img
    else:
        # Simple crop - tile is fully within image bounds
        return base_img.crop((x, y, x_end, y_end))
