# Design Document v0.2

## Overview

This document describes the design of the Ocean Floor Classifier application.

## Architecture

### Core Modules
- Project management
- Image I/O
- Grid/tile management
- Label management
- Dataset handling
- Training runs
- Inference
- Tile export

### GUI Components
- Main application window
- Labeling tab
- Training tab
- Tile viewer widget
- Tile grid widget
- Training plot widget

### CLI Components
- Project initialization
- Raw data import
- Grid operations
- Label validation
- Training
- Inference
- Tile export

## Training Modules
- Base training classes
- Keras CNN implementation
- Embedded scikit-learn implementation

## Project Folder Format (v0.2.0)

A project is a directory with the following structure:

```
<project_root>/
  project.json          # Project metadata (name, version, timestamps)
  data/
    raw/                # Raw input images
    labels.csv          # Tile labels (CSV with header: image_rel_path,tile_i,tile_j,label)
  configs/
    grid.json           # Grid specification (tile_w, tile_h, stride_x, stride_y, offset_x, offset_y, edge_policy)
    classes.json        # List of class names (JSON array)
  runs/
    train/              # Training run outputs
    infer/              # Inference run outputs
  exports/
    tiles/              # Exported tiles
```

### Project JSON Format

```json
{
  "name": "<project_name>",
  "version": "0.2.0",
  "created_utc": "<ISO8601_timestamp>",
  "updated_utc": "<ISO8601_timestamp>"
}
```

### Labels CSV Schema

The `data/labels.csv` file stores tile labels with the following schema:

- **image_rel_path**: Relative path to image from project root (posix-style, OS-agnostic)
- **tile_i**: Tile column index (integer)
- **tile_j**: Tile row index (integer)
- **label**: Label string (empty string "" for unlabeled tiles)

Rows are stored in deterministic order: sorted by `image_rel_path`, then `tile_j`, then `tile_i`. Duplicate entries for the same `(image_rel_path, tile_i, tile_j)` are not allowed.

### Grid Configuration

The `configs/grid.json` file specifies how images are tiled:

- **tile_w**: Tile width in pixels
- **tile_h**: Tile height in pixels
- **stride_x**: Horizontal stride between tiles
- **stride_y**: Vertical stride between tiles
- **offset_x**: Horizontal offset for first tile (default: 0)
- **offset_y**: Vertical offset for first tile (default: 0)
- **edge_policy**: How to handle edge tiles - "drop" (only full tiles) or "pad" (include partial tiles, caller pads)

Default grid: 256x256 tiles with 256 stride, no offset, "drop" policy.
