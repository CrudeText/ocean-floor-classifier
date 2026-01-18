"""Training run management."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from .project import OceanProject

# Import TrainingConfig with lazy import to avoid circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ofc.core.training.config import TrainingConfig


@dataclass
class TrainingHistory:
    """Training history with metrics over epochs."""

    epochs: list[int]
    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    learning_rates: Optional[list[float]] = None
    timestamps: Optional[list[str]] = None

    def to_dict(self) -> dict:
        """Convert TrainingHistory to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingHistory":
        """Create TrainingHistory from dictionary."""
        return cls(
            epochs=data["epochs"],
            train_loss=data["train_loss"],
            train_accuracy=data["train_accuracy"],
            val_loss=data["val_loss"],
            val_accuracy=data["val_accuracy"],
            learning_rates=data.get("learning_rates"),
            timestamps=data.get("timestamps"),
        )

    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Add metrics for an epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_loss: Optional validation loss
            val_accuracy: Optional validation accuracy
            learning_rate: Optional learning rate
        """
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)

        if val_loss is not None:
            self.val_loss.append(val_loss)
        else:
            self.val_loss.append(0.0)

        if val_accuracy is not None:
            self.val_accuracy.append(val_accuracy)
        else:
            self.val_accuracy.append(0.0)

        if learning_rate is not None:
            if self.learning_rates is None:
                self.learning_rates = []
            self.learning_rates.append(learning_rate)

        if self.timestamps is None:
            self.timestamps = []
        self.timestamps.append(datetime.now(timezone.utc).isoformat())


class TrainingRun:
    """
    Manages a single training run with config, checkpoints, and history.

    Each training run has its own directory under project.runs_train/
    with a unique run ID (timestamp-based).
    """

    def __init__(self, project: OceanProject, run_id: str):
        """
        Initialize training run.

        Args:
            project: OceanProject instance
            run_id: Unique run identifier (format: YYYYMMDD_HHMMSS)

        Raises:
            ValueError: If run_id format is invalid
        """
        self.project = project
        self.run_id = run_id

        # Validate run_id format (basic check)
        if not run_id or len(run_id) < 8:
            raise ValueError(f"Invalid run_id format: {run_id}")

        # Set up paths
        self.run_dir = project.paths.runs_train / run_id
        self.config_path = self.run_dir / "config.json"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.history_path = self.run_dir / "history.json"
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pth"

        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config) -> None:  # TrainingConfig type
        """
        Save training configuration to JSON file.

        Args:
            config: TrainingConfig to save
        """
        config.save(self.config_path)

    def load_config(self):  # Returns TrainingConfig
        """
        Load training configuration from JSON file.

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        from ofc.core.training.config import TrainingConfig
        return TrainingConfig.load(self.config_path)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            model: PyTorch model to save
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"

        # Load config to include in checkpoint
        from ofc.core.training.config import TrainingConfig
        config = self.load_config()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "metrics": metrics,
            "is_best": is_best,
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        torch.save(checkpoint, checkpoint_path)

        # Also save as best if indicated
        if is_best:
            torch.save(checkpoint, self.best_checkpoint_path)

        return checkpoint_path

    def load_checkpoint(
        self, epoch: Optional[int] = None, best: bool = False
    ) -> tuple[dict, int, dict]:
        """
        Load model checkpoint.

        Args:
            epoch: Specific epoch to load. If None, loads best or latest.
            best: If True, load best checkpoint instead of specific epoch

        Returns:
            Tuple of (checkpoint_dict, epoch, metrics_dict)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if best:
            checkpoint_path = self.best_checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
        elif epoch is not None:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found: {checkpoint_path}")
        else:
            # Load latest checkpoint
            checkpoints = self.list_checkpoints()
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            checkpoint_path = sorted(checkpoints)[-1]  # Latest by filename

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})

        return checkpoint, epoch, metrics

    def list_checkpoints(self) -> list[Path]:
        """
        List all checkpoint files in the run directory.

        Returns:
            List of Path objects to checkpoint files
        """
        return sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))

    def save_history(self, history: TrainingHistory) -> None:
        """
        Save training history to JSON file.

        Args:
            history: TrainingHistory to save
        """
        self.history_path.write_text(
            json.dumps(history.to_dict(), indent=2) + "\n"
        )

    def load_history(self) -> TrainingHistory:
        """
        Load training history from JSON file.

        Returns:
            TrainingHistory instance

        Raises:
            FileNotFoundError: If history file doesn't exist
            ValueError: If history file is invalid
        """
        if not self.history_path.exists():
            raise FileNotFoundError(f"History file not found: {self.history_path}")

        try:
            data = json.loads(self.history_path.read_text())
            return TrainingHistory.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in history file: {e}") from e

    def get_summary(self) -> dict:
        """
        Get summary information about this training run.

        Returns:
            Dictionary with run summary (run_id, config exists, checkpoints count, etc.)
        """
        summary = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "has_config": self.config_path.exists(),
            "checkpoint_count": len(self.list_checkpoints()),
            "has_best_checkpoint": self.best_checkpoint_path.exists(),
            "has_history": self.history_path.exists(),
        }

        # Add config info if available
        if self.config_path.exists():
            try:
                config = self.load_config()
                summary["config"] = {
                    "num_epochs": config.num_epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "optimizer": config.optimizer,
                }
            except Exception:
                pass

        # Add history info if available
        if self.history_path.exists():
            try:
                history = self.load_history()
                summary["history"] = {
                    "total_epochs": len(history.epochs),
                    "final_train_loss": history.train_loss[-1] if history.train_loss else None,
                    "final_val_loss": history.val_loss[-1] if history.val_loss else None,
                    "final_train_accuracy": history.train_accuracy[-1] if history.train_accuracy else None,
                    "final_val_accuracy": history.val_accuracy[-1] if history.val_accuracy else None,
                }
            except Exception:
                pass

        return summary


def create_run(project: OceanProject) -> TrainingRun:
    """
    Create a new training run with unique ID.

    Args:
        project: OceanProject instance

    Returns:
        TrainingRun instance with new run_id
    """
    # Generate run ID from current timestamp
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")

    # Ensure uniqueness (add suffix if needed)
    run_dir = project.paths.runs_train / run_id
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        run_dir = project.paths.runs_train / run_id

    return TrainingRun(project, run_id)


def list_runs(project: OceanProject) -> list[TrainingRun]:
    """
    List all training runs in the project.

    Args:
        project: OceanProject instance

    Returns:
        List of TrainingRun instances, sorted by run_id (newest first)
    """
    runs = []
    runs_dir = project.paths.runs_train

    if not runs_dir.exists():
        return runs

    # Find all run directories (format: YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS_XX)
    for item in runs_dir.iterdir():
        if item.is_dir():
            run_id = item.name
            # Basic validation: should match timestamp format
            if len(run_id) >= 8 and run_id.replace("_", "").replace("-", "").isdigit():
                try:
                    runs.append(TrainingRun(project, run_id))
                except Exception:
                    # Skip invalid run directories
                    continue

    # Sort by run_id (newest first)
    runs.sort(key=lambda r: r.run_id, reverse=True)

    return runs


def get_latest_run(project: OceanProject) -> Optional[TrainingRun]:
    """
    Get the most recent training run.

    Args:
        project: OceanProject instance

    Returns:
        TrainingRun instance or None if no runs exist
    """
    runs = list_runs(project)
    return runs[0] if runs else None
