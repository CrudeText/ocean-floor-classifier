"""Training related modules."""

from .auto_params import ParameterSuggester
from .config import ArchitectureConfig, LayerConfig, TrainingConfig
from .device import detect_gpu, get_device, is_gpu_available, list_available_devices
from .presets import PresetManager
from .pytorch_cnn import ConfigurableCNN, PytorchTrainer

__all__ = [
    "ArchitectureConfig",
    "LayerConfig",
    "TrainingConfig",
    "ConfigurableCNN",
    "PytorchTrainer",
    "ParameterSuggester",
    "PresetManager",
    "detect_gpu",
    "get_device",
    "is_gpu_available",
    "list_available_devices",
]
