"""Tests for grid functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from ofc.core import GridSpec


def test_gridspec_tile_rect():
    """Test tile_rect() calculates correct rectangle coordinates."""
    grid = GridSpec(
        tile_w=256, tile_h=256, stride_x=256, stride_y=256, offset_x=0, offset_y=0
    )

    # First tile
    x, y, w, h = grid.tile_rect(0, 0)
    assert x == 0
    assert y == 0
    assert w == 256
    assert h == 256

    # Second tile in x direction
    x, y, w, h = grid.tile_rect(1, 0)
    assert x == 256
    assert y == 0
    assert w == 256
    assert h == 256

    # Second tile in y direction
    x, y, w, h = grid.tile_rect(0, 1)
    assert x == 0
    assert y == 256
    assert w == 256
    assert h == 256

    # With offset
    grid_offset = GridSpec(
        tile_w=256, tile_h=256, stride_x=256, stride_y=256, offset_x=10, offset_y=20
    )
    x, y, w, h = grid_offset.tile_rect(0, 0)
    assert x == 10
    assert y == 20


def test_gridspec_tile_rect_with_stride():
    """Test tile_rect() with non-overlapping stride."""
    grid = GridSpec(
        tile_w=128, tile_h=128, stride_x=256, stride_y=256, offset_x=0, offset_y=0
    )

    x, y, w, h = grid.tile_rect(0, 0)
    assert x == 0
    assert y == 0

    x, y, w, h = grid.tile_rect(1, 0)
    assert x == 256  # Stride, not tile width
    assert y == 0


def test_gridspec_iter_tiles_drop_policy():
    """Test iter_tiles_for_image with drop policy."""
    grid = GridSpec(
        tile_w=256,
        tile_h=256,
        stride_x=256,
        stride_y=256,
        offset_x=0,
        offset_y=0,
        edge_policy="drop",
    )

    # Image exactly fits 2x2 tiles
    tiles = list(grid.iter_tiles_for_image(512, 512))
    assert len(tiles) == 4
    assert tiles[0] == (0, 0, 0, 0, 256, 256)
    assert tiles[1] == (1, 0, 256, 0, 256, 256)
    assert tiles[2] == (0, 1, 0, 256, 256, 256)
    assert tiles[3] == (1, 1, 256, 256, 256, 256)

    # Image that doesn't fit exactly - should drop edge tiles
    tiles = list(grid.iter_tiles_for_image(500, 500))
    assert len(tiles) == 4  # Still 2x2 grid
    # But check that all tiles are within bounds
    for i, j, x, y, w, h in tiles:
        assert x + w <= 500
        assert y + h <= 500

    # Image smaller than one tile
    tiles = list(grid.iter_tiles_for_image(100, 100))
    assert len(tiles) == 0  # No tiles fit


def test_gridspec_iter_tiles_pad_policy():
    """Test iter_tiles_for_image with pad policy."""
    grid = GridSpec(
        tile_w=256,
        tile_h=256,
        stride_x=256,
        stride_y=256,
        offset_x=0,
        offset_y=0,
        edge_policy="pad",
    )

    # Image exactly fits 2x2 tiles
    tiles = list(grid.iter_tiles_for_image(512, 512))
    assert len(tiles) == 4

    # Image that doesn't fit exactly - pad policy includes edge tiles
    tiles = list(grid.iter_tiles_for_image(500, 500))
    assert len(tiles) == 4  # Still includes tiles that extend beyond
    # All tiles should have full tile dimensions
    for i, j, x, y, w, h in tiles:
        assert w == 256
        assert h == 256

    # Image smaller than one tile - pad policy still includes it
    tiles = list(grid.iter_tiles_for_image(100, 100))
    assert len(tiles) == 1  # One tile that extends beyond bounds


def test_gridspec_iter_tiles_with_offset():
    """Test iter_tiles_for_image with offset."""
    grid = GridSpec(
        tile_w=256,
        tile_h=256,
        stride_x=256,
        stride_y=256,
        offset_x=50,
        offset_y=50,
        edge_policy="drop",
    )

    # Image that fits tiles with offset
    tiles = list(grid.iter_tiles_for_image(600, 600))
    assert len(tiles) >= 4
    # First tile should start at offset
    assert tiles[0][2] == 50  # x coordinate
    assert tiles[0][3] == 50  # y coordinate


def test_gridspec_to_dict_from_dict():
    """Test serialization to/from dictionary."""
    grid1 = GridSpec(
        tile_w=256,
        tile_h=128,
        stride_x=128,
        stride_y=64,
        offset_x=10,
        offset_y=20,
        edge_policy="pad",
    )

    data = grid1.to_dict()
    grid2 = GridSpec.from_dict(data)

    assert grid2.tile_w == grid1.tile_w
    assert grid2.tile_h == grid1.tile_h
    assert grid2.stride_x == grid1.stride_x
    assert grid2.stride_y == grid1.stride_y
    assert grid2.offset_x == grid1.offset_x
    assert grid2.offset_y == grid1.offset_y
    assert grid2.edge_policy == grid1.edge_policy


def test_gridspec_save_load(tmp_path: Path):
    """Test saving and loading GridSpec to/from JSON file."""
    grid1 = GridSpec(
        tile_w=256,
        tile_h=256,
        stride_x=256,
        stride_y=256,
        offset_x=0,
        offset_y=0,
        edge_policy="drop",
    )

    json_path = tmp_path / "grid.json"
    grid1.save(json_path)

    grid2 = GridSpec.load(json_path)

    assert grid2.tile_w == grid1.tile_w
    assert grid2.tile_h == grid1.tile_h
    assert grid2.stride_x == grid1.stride_x
    assert grid2.stride_y == grid1.stride_y
    assert grid2.offset_x == grid1.offset_x
    assert grid2.offset_y == grid1.offset_y
    assert grid2.edge_policy == grid1.edge_policy

    # Verify JSON content
    data = json.loads(json_path.read_text())
    assert data["tile_w"] == 256
    assert data["edge_policy"] == "drop"


def test_gridspec_default_values():
    """Test that default values are set correctly."""
    grid = GridSpec(tile_w=256, tile_h=256, stride_x=256, stride_y=256)

    assert grid.offset_x == 0
    assert grid.offset_y == 0
    assert grid.edge_policy == "drop"


def test_gridspec_invalid_edge_policy():
    """Test that invalid edge_policy raises error."""
    grid = GridSpec(
        tile_w=256,
        tile_h=256,
        stride_x=256,
        stride_y=256,
        edge_policy="invalid",  # type: ignore
    )

    with pytest.raises(ValueError, match="Unknown edge_policy"):
        list(grid.iter_tiles_for_image(512, 512))