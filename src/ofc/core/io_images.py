"""Image I/O operations."""

from collections import OrderedDict
from pathlib import Path
from typing import Optional

from PIL import Image


# Supported image extensions (case-insensitive)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    """Check if file has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def list_images(raw_dir: Path) -> list[Path]:
    """
    List all supported images in a directory (non-recursive).

    Args:
        raw_dir: Directory to search for images

    Returns:
        Sorted list of image file paths

    Raises:
        FileNotFoundError: If directory does not exist
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {raw_dir}")

    if not raw_dir.is_dir():
        raise ValueError(f"Path is not a directory: {raw_dir}")

    images = [p for p in raw_dir.iterdir() if p.is_file() and _is_image_file(p)]
    return sorted(images, key=lambda p: p.name.lower())


def read_image_pil(path: Path) -> Image.Image:
    """
    Read an image file and convert to RGB PIL Image.

    Args:
        path: Path to image file

    Returns:
        PIL Image in RGB mode

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be opened as image
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    try:
        img = Image.open(path)
        # Convert to RGB if necessary (handles RGBA, L, P, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Failed to open image {path}: {e}") from e


def get_image_size(path: Path) -> tuple[int, int]:
    """
    Get image dimensions without fully decoding the image.

    Args:
        path: Path to image file

    Returns:
        Tuple of (width, height)

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be read
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    try:
        with Image.open(path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        raise ValueError(f"Failed to read image size from {path}: {e}") from e


class TileCache:
    """Simple LRU-like cache for PIL images."""

    def __init__(self, max_items: int = 8):
        """
        Initialize cache.

        Args:
            max_items: Maximum number of images to cache
        """
        self.max_items = max_items
        # Use OrderedDict for LRU behavior (move to end on access)
        self._cache: OrderedDict[Path, Image.Image] = OrderedDict()

    def get(self, path: Path) -> Image.Image:
        """
        Get image from cache or load and cache it.

        Args:
            path: Path to image file

        Returns:
            PIL Image (cached or newly loaded)

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file cannot be opened as image
        """
        path = Path(path).resolve()

        # Check cache
        if path in self._cache:
            # Move to end (most recently used)
            img = self._cache.pop(path)
            self._cache[path] = img
            return img.copy()  # Return a copy to avoid external modifications

        # Load image
        img = read_image_pil(path)

        # Add to cache
        self._cache[path] = img

        # Evict oldest if cache is full
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)  # Remove oldest (first) item

        return img.copy()

    def clear(self) -> None:
        """Clear all cached images."""
        self._cache.clear()