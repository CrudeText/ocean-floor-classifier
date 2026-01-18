"""Tests for label functionality."""

import csv
import tempfile
from pathlib import Path

import pytest

from ofc.core import LabelsStore


def test_labels_store_get_set():
    """Test basic get/set operations."""
    store = LabelsStore()

    # Initially empty
    assert store.get("image1.jpg", 0, 0) is None

    # Set a label
    store.set("image1.jpg", 0, 0, "sand")
    assert store.get("image1.jpg", 0, 0) == "sand"

    # Update label
    store.set("image1.jpg", 0, 0, "rock")
    assert store.get("image1.jpg", 0, 0) == "rock"

    # Empty string for unlabeled
    store.set("image1.jpg", 1, 0, "")
    assert store.get("image1.jpg", 1, 0) == ""


def test_labels_store_remove():
    """Test removing labels."""
    store = LabelsStore()

    store.set("image1.jpg", 0, 0, "sand")
    assert store.get("image1.jpg", 0, 0) == "sand"

    store.remove("image1.jpg", 0, 0)
    assert store.get("image1.jpg", 0, 0) is None

    # Removing non-existent is safe
    store.remove("image1.jpg", 99, 99)


def test_labels_store_counts():
    """Test label counting."""
    store = LabelsStore()

    store.set("image1.jpg", 0, 0, "sand")
    store.set("image1.jpg", 1, 0, "sand")
    store.set("image1.jpg", 0, 1, "rock")
    store.set("image1.jpg", 1, 1, "")  # unlabeled

    counts = store.counts()
    assert counts["sand"] == 2
    assert counts["rock"] == 1
    assert counts[""] == 1


def test_labels_store_no_duplicates():
    """Test that duplicate (image, i, j) entries are not created."""
    store = LabelsStore()

    # Set same tile multiple times
    store.set("image1.jpg", 0, 0, "sand")
    store.set("image1.jpg", 0, 0, "rock")
    store.set("image1.jpg", 0, 0, "coral")

    # Should only have one entry
    counts = store.counts()
    assert counts["coral"] == 1
    assert "sand" not in counts
    assert "rock" not in counts


def test_labels_store_iter_rows():
    """Test iterating rows in deterministic order."""
    store = LabelsStore()

    # Add labels in non-sorted order
    store.set("image2.jpg", 1, 1, "rock")
    store.set("image1.jpg", 1, 0, "sand")
    store.set("image1.jpg", 0, 1, "coral")
    store.set("image1.jpg", 0, 0, "sand")
    store.set("image2.jpg", 0, 0, "rock")

    rows = list(store.iter_rows())

    # Should be sorted by image_rel_path, then j, then i
    assert len(rows) == 5
    assert rows[0] == ("image1.jpg", 0, 0, "sand")
    assert rows[1] == ("image1.jpg", 1, 0, "sand")
    assert rows[2] == ("image1.jpg", 0, 1, "coral")
    assert rows[3] == ("image2.jpg", 0, 0, "rock")
    assert rows[4] == ("image2.jpg", 1, 1, "rock")


def test_labels_store_save_load(tmp_path: Path):
    """Test saving and loading labels CSV."""
    store1 = LabelsStore()
    store1.set("image1.jpg", 0, 0, "sand")
    store1.set("image1.jpg", 1, 0, "rock")
    store1.set("image2.jpg", 0, 0, "")

    csv_path = tmp_path / "labels.csv"
    store1.save(csv_path)

    # Load back
    store2 = LabelsStore.load(csv_path)

    assert store2.get("image1.jpg", 0, 0) == "sand"
    assert store2.get("image1.jpg", 1, 0) == "rock"
    assert store2.get("image2.jpg", 0, 0) == ""

    # Verify CSV content
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3


def test_labels_store_load_creates_header_if_missing(tmp_path: Path):
    """Test that load() creates file with header if it doesn't exist."""
    csv_path = tmp_path / "labels.csv"

    store = LabelsStore.load(csv_path)

    assert csv_path.exists()
    content = csv_path.read_text()
    assert "image_rel_path,tile_i,tile_j,label" in content


def test_labels_store_save_deterministic_order(tmp_path: Path):
    """Test that save() writes rows in deterministic order."""
    store = LabelsStore()

    # Add in random order
    store.set("z_image.jpg", 0, 0, "z")
    store.set("a_image.jpg", 0, 0, "a")
    store.set("a_image.jpg", 1, 1, "b")
    store.set("a_image.jpg", 1, 0, "c")

    csv_path = tmp_path / "labels.csv"
    store.save(csv_path)

    # Read back and verify order
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Should be sorted: a_image first, then z_image
    # Within a_image: (0,0), (1,0), (1,1) - sorted by j then i
    assert rows[0]["image_rel_path"] == "a_image.jpg"
    assert rows[0]["tile_i"] == "0"
    assert rows[0]["tile_j"] == "0"
    assert rows[1]["image_rel_path"] == "a_image.jpg"
    assert rows[1]["tile_i"] == "1"
    assert rows[1]["tile_j"] == "0"
    assert rows[2]["image_rel_path"] == "a_image.jpg"
    assert rows[2]["tile_i"] == "1"
    assert rows[2]["tile_j"] == "1"
    assert rows[3]["image_rel_path"] == "z_image.jpg"


def test_labels_store_ensure_rows_for_image():
    """Test ensure_rows_for_image() adds missing rows."""
    store = LabelsStore()

    # Add some existing labels
    store.set("image1.jpg", 0, 0, "sand")
    store.set("image1.jpg", 1, 0, "rock")

    # Ensure rows for tile indices
    tile_indices = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
    store.ensure_rows_for_image("image1.jpg", tile_indices)

    # Existing labels should be preserved
    assert store.get("image1.jpg", 0, 0) == "sand"
    assert store.get("image1.jpg", 1, 0) == "rock"

    # Missing rows should be added with empty label
    assert store.get("image1.jpg", 0, 1) == ""
    assert store.get("image1.jpg", 1, 1) == ""
    assert store.get("image1.jpg", 2, 0) == ""


def test_labels_store_ensure_rows_no_duplicates():
    """Test that ensure_rows_for_image() doesn't create duplicates."""
    store = LabelsStore()

    store.set("image1.jpg", 0, 0, "sand")

    # Call ensure_rows multiple times
    store.ensure_rows_for_image("image1.jpg", [(0, 0), (1, 0)])
    store.ensure_rows_for_image("image1.jpg", [(0, 0), (1, 0)])

    # Should still have only one entry for (0,0) with original label
    assert store.get("image1.jpg", 0, 0) == "sand"
    counts = store.counts()
    assert counts["sand"] == 1


def test_labels_store_load_invalid_header(tmp_path: Path):
    """Test that load() raises error for invalid CSV header."""
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("wrong,header,columns\n")

    with pytest.raises(ValueError, match="Invalid CSV header"):
        LabelsStore.load(csv_path)