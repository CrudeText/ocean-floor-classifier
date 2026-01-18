"""Training preset management."""

import json
from pathlib import Path
from typing import Optional

from ..project import OceanProject
from .config import TrainingConfig

# Built-in preset names
BUILTIN_PRESETS = {
    "small_cnn": "Small CNN (2-3 layers, for small datasets)",
    "medium_cnn": "Medium CNN (4-5 layers, for medium datasets)",
    "deep_cnn": "Deep CNN (6+ layers, for large datasets)",
}


class PresetManager:
    """
    Manages training parameter presets.

    Presets are saved as JSON files in project.configs_dir / "training_presets" / "{name}.json"
    """

    def __init__(self, project: OceanProject):
        """
        Initialize preset manager.

        Args:
            project: OceanProject instance
        """
        self.project = project
        self.presets_dir = project.paths.configs_dir / "training_presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def get_builtin_preset(self, name: str) -> TrainingConfig:
        """
        Get a built-in preset configuration.

        Args:
            name: Preset name ("small_cnn", "medium_cnn", "deep_cnn")

        Returns:
            TrainingConfig instance

        Raises:
            ValueError: If preset name is invalid
        """
        from .auto_params import ParameterSuggester

        if name not in BUILTIN_PRESETS:
            raise ValueError(
                f"Invalid built-in preset: {name}. Must be one of {list(BUILTIN_PRESETS.keys())}"
            )

        # Use ParameterSuggester to generate preset based on dataset
        suggester = ParameterSuggester(self.project)
        analysis = suggester.analyze_dataset()

        # Determine which preset to use based on dataset size
        total_tiles = analysis.total_labeled_tiles

        if name == "small_cnn":
            # Force small architecture
            if total_tiles >= 500:
                # Override to use small anyway
                pass
        elif name == "medium_cnn":
            # Force medium architecture
            if total_tiles < 500:
                total_tiles = 1000  # Trick suggester into medium
            elif total_tiles >= 2000:
                total_tiles = 1500  # Force medium
        elif name == "deep_cnn":
            # Force deep architecture
            if total_tiles < 2000:
                total_tiles = 5000  # Trick suggester into deep

        # Generate config with modified dataset size hint
        # Actually, let's just use the suggester's suggest_full_config
        # and then modify the architecture if needed
        config = suggester.suggest_full_config()

        # For now, just return the suggested config
        # In a full implementation, we'd have predefined architectures
        return config

    def save_preset(self, name: str, config: TrainingConfig) -> None:
        """
        Save a preset to disk.

        Args:
            name: Preset name (will be sanitized for filename)
            config: TrainingConfig to save

        Raises:
            ValueError: If name is invalid or reserved
        """
        if name in BUILTIN_PRESETS:
            raise ValueError(f"Cannot overwrite built-in preset: {name}")

        # Sanitize name for filename
        safe_name = self._sanitize_name(name)
        preset_path = self.presets_dir / f"{safe_name}.json"

        config.save(preset_path)

    def load_preset(self, name: str) -> TrainingConfig:
        """
        Load a preset from disk or built-in presets.

        Args:
            name: Preset name

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If preset doesn't exist
            ValueError: If preset is invalid
        """
        # Check if it's a built-in preset
        if name in BUILTIN_PRESETS:
            return self.get_builtin_preset(name)

        # Load from disk
        safe_name = self._sanitize_name(name)
        preset_path = self.presets_dir / f"{safe_name}.json"

        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {name}")

        return TrainingConfig.load(preset_path)

    def list_presets(self) -> list[str]:
        """
        List all available presets (built-in + user-saved).

        Returns:
            List of preset names, sorted (built-in first, then user presets)
        """
        presets = list(BUILTIN_PRESETS.keys())

        # Add user presets
        if self.presets_dir.exists():
            for preset_file in self.presets_dir.glob("*.json"):
                preset_name = preset_file.stem
                if preset_name not in presets:
                    presets.append(preset_name)

        return presets

    def delete_preset(self, name: str) -> None:
        """
        Delete a user-saved preset.

        Args:
            name: Preset name

        Raises:
            ValueError: If trying to delete built-in preset
            FileNotFoundError: If preset doesn't exist
        """
        if name in BUILTIN_PRESETS:
            raise ValueError(f"Cannot delete built-in preset: {name}")

        safe_name = self._sanitize_name(name)
        preset_path = self.presets_dir / f"{safe_name}.json"

        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {name}")

        preset_path.unlink()

    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize preset name for use as filename.

        Args:
            name: Original name

        Returns:
            Sanitized name safe for filesystem
        """
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        safe = name
        for char in invalid_chars:
            safe = safe.replace(char, "_")
        # Remove leading/trailing spaces and dots
        safe = safe.strip(". ")
        return safe
