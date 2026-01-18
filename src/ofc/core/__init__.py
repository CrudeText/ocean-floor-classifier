"""Core application logic."""

from .dataset import (
    DatasetStats,
    TrainingDataset,
    get_class_weights,
    get_dataset_stats,
    get_train_val_split,
)
from .grid import GridSpec
from .io_images import TileCache, get_image_size, list_images, read_image_pil
from .labels import LabelsStore, ensure_image_tiles
from .project import OceanProject, ProjectPaths
from .runs import TrainingHistory, TrainingRun, create_run, get_latest_run, list_runs
from .sample_data import (
    auto_generate_test_labels_if_needed,
    generate_labels_for_all_images,
    generate_test_labels,
)
from .tiles import TileRef, enumerate_tiles_for_image, get_tile_image

__all__ = [
    "OceanProject",
    "ProjectPaths",
    "GridSpec",
    "LabelsStore",
    "TileRef",
    "enumerate_tiles_for_image",
    "get_tile_image",
    "list_images",
    "get_image_size",
    "TileCache",
    "ensure_image_tiles",
    "read_image_pil",
    "TrainingDataset",
    "DatasetStats",
    "get_train_val_split",
    "get_class_weights",
    "get_dataset_stats",
    "TrainingRun",
    "TrainingHistory",
    "create_run",
    "list_runs",
    "get_latest_run",
    "generate_test_labels",
    "auto_generate_test_labels_if_needed",
    "generate_labels_for_all_images",
]
