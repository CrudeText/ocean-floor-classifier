"""Training configuration classes."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class LayerConfig:
    """Configuration for a single neural network layer."""

    layer_type: str  # "conv2d", "maxpool2d", "avgpool2d", "linear", "dropout", "batchnorm2d"
    params: dict = field(default_factory=dict)  # Layer-specific parameters
    activation: Optional[str] = None  # "relu", "sigmoid", "tanh", "softmax", None

    def to_dict(self) -> dict:
        """Convert LayerConfig to dictionary."""
        return {
            "layer_type": self.layer_type,
            "params": self.params,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LayerConfig":
        """Create LayerConfig from dictionary."""
        return cls(
            layer_type=data["layer_type"],
            params=data.get("params", {}),
            activation=data.get("activation"),
        )

    def validate(self) -> None:
        """
        Validate layer configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        valid_types = {
            "conv2d",
            "maxpool2d",
            "avgpool2d",
            "linear",
            "dropout",
            "batchnorm2d",
        }
        if self.layer_type not in valid_types:
            raise ValueError(
                f"Invalid layer_type: {self.layer_type}. "
                f"Must be one of {valid_types}"
            )

        valid_activations = {"relu", "sigmoid", "tanh", "softmax", None}
        if self.activation not in valid_activations:
            raise ValueError(
                f"Invalid activation: {self.activation}. "
                f"Must be one of {valid_activations}"
            )

        # Validate layer-specific parameters
        if self.layer_type == "conv2d":
            required = ["out_channels"]
            for param in required:
                if param not in self.params:
                    raise ValueError(f"conv2d layer missing required parameter: {param}")
            if "kernel_size" not in self.params:
                self.params["kernel_size"] = 3  # Default
            if "stride" not in self.params:
                self.params["stride"] = 1  # Default
            if "padding" not in self.params:
                self.params["padding"] = 1  # Default

        elif self.layer_type == "linear":
            if "out_features" not in self.params:
                raise ValueError("linear layer missing required parameter: out_features")

        elif self.layer_type == "dropout":
            if "p" not in self.params:
                self.params["p"] = 0.5  # Default

        elif self.layer_type in ("maxpool2d", "avgpool2d"):
            if "kernel_size" not in self.params:
                self.params["kernel_size"] = 2  # Default
            if "stride" not in self.params:
                self.params["stride"] = 2  # Default


@dataclass
class ArchitectureConfig:
    """Configuration for neural network architecture."""

    input_size: tuple[int, int]  # (width, height)
    layers: list[LayerConfig]
    num_classes: int

    def to_dict(self) -> dict:
        """Convert ArchitectureConfig to dictionary."""
        return {
            "input_size": list(self.input_size),  # Convert tuple to list for JSON
            "layers": [layer.to_dict() for layer in self.layers],
            "num_classes": self.num_classes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureConfig":
        """Create ArchitectureConfig from dictionary."""
        return cls(
            input_size=tuple(data["input_size"]),  # Convert list back to tuple
            layers=[LayerConfig.from_dict(layer_data) for layer_data in data["layers"]],
            num_classes=data["num_classes"],
        )

    def validate(self) -> None:
        """
        Validate architecture configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if len(self.input_size) != 2:
            raise ValueError(f"input_size must be (width, height), got {self.input_size}")
        width, height = self.input_size
        if width <= 0 or height <= 0:
            raise ValueError(f"input_size dimensions must be positive, got {self.input_size}")

        if not self.layers:
            raise ValueError("layers list cannot be empty")

        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")

        # Validate each layer
        for i, layer in enumerate(self.layers):
            try:
                layer.validate()
            except ValueError as e:
                raise ValueError(f"Layer {i} validation failed: {e}") from e


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    architecture: ArchitectureConfig
    batch_size: int
    num_epochs: int
    learning_rate: float
    optimizer: str  # "adam", "sgd", "rmsprop", "adamw"
    optimizer_params: dict = field(default_factory=dict)  # weight_decay, momentum, etc.
    loss_function: str = "cross_entropy"  # "cross_entropy", "focal_loss", etc.
    use_gpu: bool = True
    device: Optional[str] = None  # Specific device string (e.g., "cuda:0", "mps", "cpu")
    validation_split: float = 0.2
    data_augmentation: dict = field(
        default_factory=dict
    )  # flip, rotate, brightness, etc.
    early_stopping: Optional[dict] = None  # {"patience": int, "min_delta": float}
    class_weights: bool = False  # Auto-balance classes
    seed: Optional[int] = None
    save_checkpoint_every: int = 5  # Save checkpoint every N epochs

    def to_dict(self) -> dict:
        """Convert TrainingConfig to dictionary."""
        return {
            "architecture": self.architecture.to_dict(),
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "optimizer_params": self.optimizer_params,
            "loss_function": self.loss_function,
            "use_gpu": self.use_gpu,
            "device": self.device,
            "validation_split": self.validation_split,
            "data_augmentation": self.data_augmentation,
            "early_stopping": self.early_stopping,
            "class_weights": self.class_weights,
            "seed": self.seed,
            "save_checkpoint_every": self.save_checkpoint_every,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """Create TrainingConfig from dictionary."""
        return cls(
            architecture=ArchitectureConfig.from_dict(data["architecture"]),
            batch_size=data["batch_size"],
            num_epochs=data["num_epochs"],
            learning_rate=data["learning_rate"],
            optimizer=data["optimizer"],
            optimizer_params=data.get("optimizer_params", {}),
            loss_function=data.get("loss_function", "cross_entropy"),
            use_gpu=data.get("use_gpu", True),
            device=data.get("device"),
            validation_split=data.get("validation_split", 0.2),
            data_augmentation=data.get("data_augmentation", {}),
            early_stopping=data.get("early_stopping"),
            class_weights=data.get("class_weights", False),
            seed=data.get("seed"),
            save_checkpoint_every=data.get("save_checkpoint_every", 5),
        )

    def validate(self) -> None:
        """
        Validate training configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate architecture
        try:
            self.architecture.validate()
        except ValueError as e:
            raise ValueError(f"Architecture validation failed: {e}") from e

        # Validate hyperparameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        valid_optimizers = {"adam", "sgd", "rmsprop", "adamw"}
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer: {self.optimizer}. Must be one of {valid_optimizers}"
            )

        valid_losses = {"cross_entropy", "focal_loss", "nll_loss"}
        if self.loss_function not in valid_losses:
            raise ValueError(
                f"Invalid loss_function: {self.loss_function}. "
                f"Must be one of {valid_losses}"
            )

        if not 0.0 <= self.validation_split <= 0.5:
            raise ValueError(
                f"validation_split must be in [0, 0.5], got {self.validation_split}"
            )

        if self.save_checkpoint_every <= 0:
            raise ValueError(
                f"save_checkpoint_every must be positive, got {self.save_checkpoint_every}"
            )

        # Validate early stopping if provided
        if self.early_stopping is not None:
            if "patience" not in self.early_stopping:
                raise ValueError("early_stopping must include 'patience'")
            if self.early_stopping["patience"] <= 0:
                raise ValueError(
                    f"early_stopping patience must be positive, "
                    f"got {self.early_stopping['patience']}"
                )

    def save(self, path: str | Path) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            data = json.loads(path.read_text())
            config = cls.from_dict(data)
            config.validate()
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e
        except KeyError as e:
            raise ValueError(f"Missing required field in config: {e}") from e
