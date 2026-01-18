"""Automatic parameter suggestion system."""

from typing import Optional

from ..dataset import DatasetStats, get_dataset_stats
from ..grid import GridSpec
from ..labels import LabelsStore
from ..project import OceanProject
from .config import ArchitectureConfig, LayerConfig, TrainingConfig


class ParameterSuggester:
    """
    Suggests optimal training parameters based on dataset analysis.

    Analyzes dataset characteristics (size, classes, distribution) and
    suggests appropriate architecture and hyperparameters.
    """

    def __init__(self, project: OceanProject):
        """
        Initialize parameter suggester.

        Args:
            project: OceanProject instance to analyze
        """
        self.project = project

    def analyze_dataset(self) -> DatasetStats:
        """
        Analyze the dataset and return statistics.

        Returns:
            DatasetStats object with dataset characteristics

        Raises:
            ValueError: If project structure is invalid
        """
        grid = self.project.get_grid()
        labels = self.project.get_labels()

        return get_dataset_stats(self.project, grid, labels)

    def suggest_architecture(
        self, analysis: DatasetStats, input_size: Optional[tuple[int, int]] = None
    ) -> ArchitectureConfig:
        """
        Suggest neural network architecture based on dataset analysis.

        Args:
            analysis: DatasetStats from analyze_dataset()
            input_size: Optional (width, height) tuple. If None, uses grid tile size.

        Returns:
            ArchitectureConfig with suggested architecture
        """
        # Determine input size
        if input_size is None:
            grid = self.project.get_grid()
            input_size = (grid.tile_w, grid.tile_h)

        # Determine architecture complexity based on dataset size
        total_tiles = analysis.total_labeled_tiles
        num_classes = analysis.num_classes

        layers = []

        if total_tiles < 500:
            # Small dataset: Simple 2-3 layer CNN
            layers.extend([
                LayerConfig(
                    "conv2d",
                    {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                    activation="relu",
                ),
                LayerConfig("maxpool2d", {"kernel_size": 2, "stride": 2}),
                LayerConfig(
                    "conv2d",
                    {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                    activation="relu",
                ),
                LayerConfig("maxpool2d", {"kernel_size": 2, "stride": 2}),
                LayerConfig("dropout", {"p": 0.25}),
            ])

        elif total_tiles < 2000:
            # Medium dataset: 4-5 layer CNN
            layers.extend([
                LayerConfig(
                    "conv2d",
                    {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                    activation="relu",
                ),
                LayerConfig("batchnorm2d", {}),
                LayerConfig("maxpool2d", {"kernel_size": 2, "stride": 2}),
                LayerConfig(
                    "conv2d",
                    {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                    activation="relu",
                ),
                LayerConfig("batchnorm2d", {}),
                LayerConfig("maxpool2d", {"kernel_size": 2, "stride": 2}),
                LayerConfig(
                    "conv2d",
                    {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1},
                    activation="relu",
                ),
                LayerConfig("dropout", {"p": 0.3}),
            ])

        else:
            # Large dataset: Deeper 6+ layer CNN
            layers.extend([
                LayerConfig(
                    "conv2d",
                    {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                    activation="relu",
                ),
                LayerConfig("batchnorm2d", {}),
                LayerConfig("maxpool2d", {"kernel_size": 2, "stride": 2}),
                LayerConfig(
                    "conv2d",
                    {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1},
                    activation="relu",
                ),
                LayerConfig("batchnorm2d", {}),
                LayerConfig("maxpool2d", {"kernel_size": 2, "stride": 2}),
                LayerConfig(
                    "conv2d",
                    {"out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1},
                    activation="relu",
                ),
                LayerConfig("batchnorm2d", {}),
                LayerConfig("maxpool2d", {"kernel_size": 2, "stride": 2}),
                LayerConfig(
                    "conv2d",
                    {"out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1},
                    activation="relu",
                ),
                LayerConfig("dropout", {"p": 0.4}),
            ])

        return ArchitectureConfig(
            input_size=input_size,
            layers=layers,
            num_classes=num_classes,
        )

    def suggest_hyperparameters(
        self, analysis: DatasetStats, use_gpu: bool = True
    ) -> dict:
        """
        Suggest training hyperparameters based on dataset analysis.

        Args:
            analysis: DatasetStats from analyze_dataset()
            use_gpu: Whether GPU will be used (affects batch size)

        Returns:
            Dictionary with suggested hyperparameters
        """
        total_tiles = analysis.total_labeled_tiles
        num_classes = analysis.num_classes

        # Batch size: depends on GPU availability and dataset size
        if use_gpu:
            # GPU: larger batches
            if total_tiles < 500:
                batch_size = 16
            elif total_tiles < 2000:
                batch_size = 32
            elif total_tiles < 10000:
                batch_size = 64
            else:
                batch_size = 128
        else:
            # CPU: smaller batches
            if total_tiles < 500:
                batch_size = 8
            elif total_tiles < 2000:
                batch_size = 16
            else:
                batch_size = 32

        # Learning rate: adaptive based on dataset size
        if total_tiles < 500:
            learning_rate = 1e-3  # Smaller dataset, higher LR
        elif total_tiles < 2000:
            learning_rate = 5e-4
        elif total_tiles < 10000:
            learning_rate = 1e-4
        else:
            learning_rate = 5e-5  # Large dataset, lower LR

        # Number of epochs: more for smaller datasets
        if total_tiles < 500:
            num_epochs = 50
        elif total_tiles < 2000:
            num_epochs = 30
        elif total_tiles < 10000:
            num_epochs = 20
        else:
            num_epochs = 15

        # Optimizer: Adam is good default, SGD for very large datasets
        if total_tiles > 50000:
            optimizer = "sgd"
            optimizer_params = {"momentum": 0.9, "weight_decay": 1e-4}
        else:
            optimizer = "adam"
            optimizer_params = {"weight_decay": 1e-5}

        # Validation split: standard 20%
        validation_split = 0.2

        # Class weights: suggest if imbalance detected
        class_weights = False
        if num_classes > 1:
            # Check for class imbalance (if any class has < 10% of samples)
            min_ratio = min(
                count / analysis.total_labeled_tiles
                for count in analysis.class_counts.values()
            )
            if min_ratio < 0.1:
                class_weights = True

        # Early stopping: enable for smaller datasets
        early_stopping = None
        if total_tiles < 2000:
            early_stopping = {"patience": 10, "min_delta": 0.001}

        # Data augmentation: no defaults (user must explicitly enable)
        data_augmentation = {}

        return {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "optimizer": optimizer,
            "optimizer_params": optimizer_params,
            "validation_split": validation_split,
            "class_weights": class_weights,
            "early_stopping": early_stopping,
            "data_augmentation": data_augmentation,
        }

    def suggest_full_config(
        self, use_gpu: bool = True, input_size: Optional[tuple[int, int]] = None
    ) -> TrainingConfig:
        """
        Suggest complete training configuration.

        Combines architecture and hyperparameter suggestions into a full
        TrainingConfig ready for training.

        Args:
            use_gpu: Whether GPU will be used
            input_size: Optional (width, height) tuple. If None, uses grid tile size.

        Returns:
            TrainingConfig with all suggested parameters

        Raises:
            ValueError: If project has no labeled data or invalid structure
        """
        # Analyze dataset
        analysis = self.analyze_dataset()

        if analysis.total_labeled_tiles == 0:
            raise ValueError(
                "Cannot suggest parameters: no labeled tiles found in dataset"
            )

        if analysis.num_classes == 0:
            raise ValueError(
                "Cannot suggest parameters: no classes found in dataset"
            )

        # Suggest architecture
        architecture = self.suggest_architecture(analysis, input_size)

        # Suggest hyperparameters
        hyperparams = self.suggest_hyperparameters(analysis, use_gpu)

        # Create full config
        config = TrainingConfig(
            architecture=architecture,
            batch_size=hyperparams["batch_size"],
            num_epochs=hyperparams["num_epochs"],
            learning_rate=hyperparams["learning_rate"],
            optimizer=hyperparams["optimizer"],
            optimizer_params=hyperparams["optimizer_params"],
            loss_function="cross_entropy",
            use_gpu=use_gpu,
            validation_split=hyperparams["validation_split"],
            data_augmentation=hyperparams["data_augmentation"],
            early_stopping=hyperparams["early_stopping"],
            class_weights=hyperparams["class_weights"],
            seed=42,  # Default seed for reproducibility
            save_checkpoint_every=5,
        )

        # Validate config
        config.validate()

        return config
