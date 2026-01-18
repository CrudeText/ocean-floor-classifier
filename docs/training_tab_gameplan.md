# Training Tab Implementation - Game Plan v2.0

## Technology Stack Decisions

- **Framework**: PyTorch (replaces TensorFlow)
- **Graphing**: PyQtGraph (real-time plotting)
- **Architecture**: Fully configurable (user controls all layers)
- **Checkpoint Format**: PyTorch `.pth` files + separate JSON config files
- **GPU Support**: Full CUDA/ROCm/MPS support with user toggle

---

## Phase 1: Core Training Infrastructure (Backend)

### Step 1.1: Update Dependencies
- Replace `tensorflow` with `torch` and `torchvision` in `pyproject.toml`
- Add `pyqtgraph` to dependencies
- Note: User will need to install PyTorch with CUDA support separately if they want GPU

### Step 1.2: Dataset Loading Module (`src/ofc/core/dataset.py`)
**Purpose**: Load labeled tiles from project and prepare for training

**Classes/Functions**:
- `TrainingDataset` class (inherits `torch.utils.data.Dataset`):
  - `__init__(project: OceanProject, grid: GridSpec, labels: LabelsStore, transform=None)`
  - `__len__() -> int`
  - `__getitem__(idx) -> (tensor, label_idx)`
  - Loads tile images, applies transforms, converts labels to class indices
- `get_train_val_split(dataset, val_split=0.2, seed=None) -> (train_dataset, val_dataset)`
- `get_class_weights(dataset) -> torch.Tensor` (for imbalanced datasets)
- `get_dataset_stats(project) -> dict` (size, classes, distribution)

**Data Flow**:
- Read `labels.csv` from project
- Filter rows with non-empty labels
- Load tiles using `get_tile_image()` from `tiles.py`
- Convert PIL images to PyTorch tensors
- Map class names to indices

### Step 1.3: Architecture Configuration System (`src/ofc/core/training/config.py`)
**Purpose**: Define and validate training configurations

**Classes**:
- `ArchitectureConfig` dataclass:
  - `input_size: tuple[int, int]` (width, height)
  - `layers: list[LayerConfig]` (ordered list of layer definitions)
  - `num_classes: int`
  
- `LayerConfig` dataclass:
  - `layer_type: str` ("conv2d", "maxpool2d", "avgpool2d", "linear", "dropout", "batchnorm2d")
  - `params: dict` (layer-specific parameters)
    - For conv2d: `out_channels`, `kernel_size`, `stride`, `padding`
    - For linear: `out_features`
    - For dropout: `p`
    - etc.
  - `activation: str | None` ("relu", "sigmoid", "tanh", "softmax", None)
  
- `TrainingConfig` dataclass:
  - `architecture: ArchitectureConfig`
  - `batch_size: int`
  - `num_epochs: int`
  - `learning_rate: float`
  - `optimizer: str` ("adam", "sgd", "rmsprop", "adamw")
  - `optimizer_params: dict` (weight_decay, momentum, etc.)
  - `loss_function: str` ("cross_entropy", "focal_loss", etc.)
  - `use_gpu: bool`
  - `validation_split: float` (0.0 to 0.5)
  - `data_augmentation: dict` (flip, rotate, brightness, etc.)
  - `early_stopping: dict | None` (patience, min_delta)
  - `class_weights: bool` (auto-balance)
  - `seed: int | None`
  - `save_checkpoint_every: int` (epochs)
  
- Methods:
  - `to_dict() -> dict` (for JSON export)
  - `from_dict(d: dict) -> TrainingConfig` (for JSON import)
  - `validate() -> None` (raises ValueError if invalid)

### Step 1.4: Model Builder (`src/ofc/core/training/pytorch_cnn.py`)
**Purpose**: Build PyTorch models from architecture configs

**Classes**:
- `ConfigurableCNN(nn.Module)`:
  - `__init__(config: ArchitectureConfig)`
  - `forward(x) -> tensor`
  - Dynamically builds model from layer configs
  
- `PytorchTrainer`:
  - `__init__(config: TrainingConfig, project: OceanProject)`
  - `train() -> TrainingHistory`
  - `validate() -> dict` (metrics)
  - `save_checkpoint(path, epoch, metrics)`
  - `load_checkpoint(path) -> (model, epoch, metrics)`
  - GPU detection and device management
  - Training loop with callbacks for metrics

**Training Loop Features**:
- Move model and data to correct device (CPU/GPU)
- Progress callbacks (emit metrics per epoch/batch)
- Checkpoint saving
- Early stopping
- Learning rate scheduling (optional)

### Step 1.5: Parameter Suggestion System (`src/ofc/core/training/auto_params.py`)
**Purpose**: Suggest optimal parameters based on dataset analysis

**Classes**:
- `ParameterSuggester`:
  - `__init__(project: OceanProject)`
  - `analyze_dataset() -> DatasetAnalysis`:
    - Count labeled tiles
    - Count classes and distribution
    - Calculate average image size
    - Detect class imbalance
    - Estimate dataset complexity
  
  - `suggest_architecture(analysis: DatasetAnalysis) -> ArchitectureConfig`:
    - Small dataset (<1000 tiles): Simple 2-3 layer CNN
    - Medium (1000-10000): 4-5 layer CNN
    - Large (>10000): Deeper CNN (6+ layers)
    - Adjust input size based on tile dimensions
    - Suggest dropout based on dataset size
  
  - `suggest_hyperparameters(analysis: DatasetAnalysis, use_gpu: bool) -> dict`:
    - Batch size: 
      - GPU available: 32-128 (based on GPU memory)
      - CPU only: 16-32
    - Learning rate: 1e-3 to 1e-4 (adaptive based on dataset size)
    - Epochs: 20-50 (more for smaller datasets)
    - Optimizer: Adam (default), SGD for very large datasets
    - Validation split: 0.2 (default)
  
  - `suggest_full_config(use_gpu: bool = True) -> TrainingConfig`:
    - Combines architecture + hyperparameters
    - Returns complete `TrainingConfig`

**Heuristics**:
- Dataset size → model complexity
- Class count → output layer size
- Class imbalance → suggest class weights
- GPU availability → batch size limits

### Step 1.6: Training Run Management (`src/ofc/core/runs.py`)
**Purpose**: Manage training runs, save/load configs and checkpoints

**Classes**:
- `TrainingRun`:
  - `__init__(project: OceanProject, run_id: str | None = None)`
  - `run_id: str` (timestamp-based: `YYYYMMDD_HHMMSS`)
  - `run_dir: Path` (project.runs_train / run_id)
  - `config_path: Path` (run_dir / "config.json")
  - `checkpoint_dir: Path` (run_dir / "checkpoints")
  - `history_path: Path` (run_dir / "history.json")
  
  - Methods:
    - `save_config(config: TrainingConfig)`
    - `load_config() -> TrainingConfig`
    - `save_checkpoint(model, epoch, metrics, is_best=False)`
    - `load_checkpoint(epoch: int | None = None) -> (model, epoch, metrics)`
    - `save_history(history: TrainingHistory)`
    - `load_history() -> TrainingHistory`
    - `list_checkpoints() -> list[Path]`

- `TrainingHistory` dataclass:
  - `epochs: list[int]`
  - `train_loss: list[float]`
  - `train_accuracy: list[float]`
  - `val_loss: list[float]`
  - `val_accuracy: list[float]`
  - `learning_rates: list[float]` (optional)
  - `timestamps: list[str]`

- Functions:
  - `create_run(project: OceanProject) -> TrainingRun`
  - `list_runs(project: OceanProject) -> list[TrainingRun]`
  - `get_latest_run(project: OceanProject) -> TrainingRun | None`

### Step 1.7: Preset Management (`src/ofc/core/training/presets.py`)
**Purpose**: Define and manage parameter presets

**Classes**:
- `PresetManager`:
  - Built-in presets:
    - `"small_cnn"`: 2-3 conv layers, small batch, for small datasets
    - `"medium_cnn"`: 4-5 conv layers, medium batch
    - `"deep_cnn"`: 6+ conv layers, larger batch
    - `"resnet_style"`: ResNet-like architecture (if we implement)
    - `"custom"`: User-defined
  
  - Methods:
    - `get_preset(name: str) -> TrainingConfig`
    - `save_preset(name: str, config: TrainingConfig, project: OceanProject)`
    - `load_preset(name: str, project: OceanProject) -> TrainingConfig`
    - `list_presets(project: OceanProject) -> list[str]`
    - `delete_preset(name: str, project: OceanProject)`
  
  - Storage: `project.configs_dir / "training_presets" / "{name}.json"`

---

## Phase 2: Training Tab UI - Parameters Section

### Step 2.1: Basic Layout Structure (`src/ofc/gui/tabs/train_tab.py`)
**Layout**:
```
┌─────────────────────────────────────────┐
│  Training Tab                           │
├─────────────────────────────────────────┤
│  [Parameters Section - Scrollable]     │
│  ┌───────────────────────────────────┐ │
│  │ Model Architecture                │ │
│  │ Preset: [Dropdown ▼] [Load]      │ │
│  │ [Export] [Import] [Auto-Suggest] │ │
│  └───────────────────────────────────┘ │
│  ┌───────────────────────────────────┐ │
│  │ Architecture Config              │ │
│  │ ...                               │ │
│  └───────────────────────────────────┘ │
│  ┌───────────────────────────────────┐ │
│  │ Training Hyperparameters          │ │
│  │ ...                               │ │
│  └───────────────────────────────────┘ │
│  ┌───────────────────────────────────┐ │
│  │ Advanced Options                  │ │
│  │ ...                               │ │
│  └───────────────────────────────────┘ │
│  ┌───────────────────────────────────┐ │
│  │ Data Selection                    │ │
│  │ ...                               │ │
│  └───────────────────────────────────┘ │
├─────────────────────────────────────────┤
│  [Graphs Section]                       │
│  ┌──────────────┬──────────────┐      │
│  │ Loss Plot    │ Accuracy Plot │      │
│  └──────────────┴──────────────┘      │
├─────────────────────────────────────────┤
│  [Control Buttons]                      │
│  [Start Training] [Stop] [Save Model]  │
│  Status: [Epoch X/Y, Loss: ...]        │
└─────────────────────────────────────────┘
```

### Step 2.2: Preset Management UI
- **Preset Dropdown**: Shows available presets (built-in + user-saved)
- **Load Button**: Loads selected preset into all parameter fields
- **Export Button**: Exports current parameters to JSON file (file dialog)
- **Import Button**: Imports parameters from JSON file (file dialog)
- **Auto-Suggest Button**: Calls `ParameterSuggester.suggest_full_config()` and populates fields

### Step 2.3: Model Architecture Parameters
**Group Box: "Model Architecture"**

- **Input Size**:
  - Width: SpinBox (64-2048, default: 256)
  - Height: SpinBox (64-2048, default: 256)
  
- **Layers List** (Dynamic):
  - Scrollable list/table of layers
  - Each row: [Type ▼] [Parameters...] [Activation ▼] [↑] [↓] [Remove]
  - **Add Layer** button (adds new layer row)
  - Layer types: Conv2D, MaxPool2D, AvgPool2D, Linear, Dropout, BatchNorm2D
  
- **Layer Parameter Editor** (when layer selected):
  - Dynamic form based on layer type
  - Conv2D: Out Channels, Kernel Size, Stride, Padding
  - Linear: Out Features
  - Dropout: Probability
  - Pool: Kernel Size, Stride

### Step 2.4: Training Hyperparameters
**Group Box: "Training Parameters"**

- Batch Size: SpinBox (1-512, default: 32)
- Number of Epochs: SpinBox (1-1000, default: 20)
- Learning Rate: DoubleSpinBox (1e-6 to 1.0, scientific notation, default: 0.001)
- Optimizer: Dropdown (Adam, SGD, RMSprop, AdamW)
- Optimizer Parameters (expandable):
  - Weight Decay: DoubleSpinBox
  - Momentum: DoubleSpinBox (for SGD)
- Loss Function: Dropdown (Cross Entropy, Focal Loss, etc.)
- Validation Split: Slider (0-50%, default: 20%)

### Step 2.5: Advanced Options
**Group Box: "Advanced Options"**

- **Device Selection**:
  - "Use GPU" Checkbox (enabled if GPU available, shows GPU name)
  - GPU Info Label: "GPU: NVIDIA RTX 3080 (8GB)" or "No GPU available"
  
- **Data Augmentation** (checkboxes):
  - Horizontal Flip
  - Vertical Flip
  - Random Rotation (±degrees)
  - Brightness Adjustment
  - Contrast Adjustment
  
- **Training Options**:
  - Early Stopping: Checkbox
    - Patience: SpinBox (epochs)
    - Min Delta: DoubleSpinBox
  - Class Weights: Checkbox ("Auto-balance classes")
  - Seed: SpinBox (for reproducibility, optional)
  - Save Checkpoint Every: SpinBox (epochs, default: 5)

### Step 2.6: Data Selection
**Group Box: "Data Selection"**

- **Dataset Info** (read-only):
  - Total Labeled Tiles: [count]
  - Classes: [list with counts]
  - Class Distribution: [simple bar chart or text]
  
- **Filters** (optional):
  - Exclude classes: [multi-select]
  - Min tiles per class: [spinbox]

---

## Phase 3: Real-time Monitoring (Graphs)

### Step 3.1: PyQtGraph Integration (`src/ofc/gui/widgets/training_plot.py`)
**Purpose**: Real-time training metrics visualization

**Classes**:
- `TrainingPlotWidget(QWidget)`:
  - `__init__()`
  - `add_loss_plot()`: Creates loss subplot (train + validation)
  - `add_accuracy_plot()`: Creates accuracy subplot
  - `update_metrics(epoch, train_loss, val_loss, train_acc, val_acc)`
  - `clear()`: Reset plots
  - `set_max_epochs(max_epochs)`: Set x-axis range

**Features**:
- Two subplots side-by-side (Loss, Accuracy)
- Real-time line updates (append data points)
- Legend (Train, Validation)
- Auto-scaling axes
- Grid lines
- Colors: Train=blue, Validation=orange

### Step 3.2: Graph Layout in Train Tab
- Add `TrainingPlotWidget` to TrainTab
- Position: Middle section, below parameters
- Size: Minimum 400px height, expandable
- Update frequency: Every epoch (or every N batches for smoother curves)

### Step 3.3: Metrics Collection & Signals
**Signal/Slot System**:
- `TrainingThread` emits Qt signals:
  - `epoch_completed(int, dict)` (epoch number, metrics dict)
  - `batch_completed(int, dict)` (optional, for smoother updates)
  - `training_finished(TrainingHistory)`
  - `training_error(str)` (error message)
  
- `TrainTab` connects signals to:
  - Graph updates
  - Status label updates
  - Progress bar updates

---

## Phase 4: Training Execution & Threading

### Step 4.1: Training Thread (`src/ofc/gui/tabs/train_tab.py`)
**Class**: `TrainingThread(QThread)`

**Methods**:
- `__init__(config: TrainingConfig, project: OceanProject, run: TrainingRun)`
- `run()`: Main training loop (calls `PytorchTrainer.train()`)
- `stop()`: Graceful cancellation (sets flag, trainer checks periodically)

**Signals**:
- `epoch_completed(int, dict)` (epoch, metrics)
- `training_finished(TrainingHistory)`
- `training_error(str)`
- `status_update(str)` (status messages)

### Step 4.2: Run Button & Controls
**Buttons**:
- **Start Training**: 
  - Validates all parameters
  - Creates `TrainingRun`
  - Starts `TrainingThread`
  - Disables parameter inputs
  - Changes to "Training..." (disabled)
  
- **Stop Training**:
  - Enabled during training
  - Calls `thread.stop()`
  - Saves current checkpoint
  
- **Save Model**:
  - Enabled during/after training
  - Saves current checkpoint to user-selected location
  - Or saves to run directory

**Status Display**:
- Status Label: "Ready" / "Training epoch 5/20..." / "Completed"
- Progress Bar: Epoch progress (0-100%)
- Metrics Display: Current loss, accuracy (updated in real-time)

### Step 4.3: Parameter Validation
**Validation Checks**:
- Project has labeled data
- Classes are defined (non-empty)
- Architecture layers are valid (output sizes match)
- Batch size > 0
- Learning rate > 0
- Validation split in range [0, 0.5]
- GPU available if "Use GPU" is checked

**Error Handling**:
- Show `QMessageBox` with error details
- Highlight invalid fields (red border)
- Prevent training start if invalid

### Step 4.4: GPU Detection & Display
**GPU Detection Utility** (`src/ofc/core/training/device.py`):
- `detect_gpu() -> dict | None`:
  - Returns `{"name": "NVIDIA RTX 3080", "memory_gb": 8}` or `None`
- `is_gpu_available() -> bool`
- `get_device(use_gpu: bool) -> torch.device`

**UI Integration**:
- Check GPU on tab load
- Update "Use GPU" checkbox state
- Show GPU info in Advanced Options
- Warn if GPU requested but unavailable

---

## Phase 5: Integration & Polish

### Step 5.1: Project Integration
- `TrainTab.set_project(project: OceanProject)`:
  - Loads project data
  - Updates dataset info
  - Loads available presets
  - Shows previous runs (optional: dropdown)

### Step 5.2: Checkpoint Management UI
- **Checkpoint List**: Shows saved checkpoints in current run
- **Load Checkpoint**: Button to load weights into model
- **Resume Training**: Option to continue from checkpoint

### Step 5.3: Export & Results
- **Export Model**: Save model weights + config to external location
- **Training Summary**: Dialog showing final metrics, training time, etc.
- **Export History**: Save training history as CSV/JSON

### Step 5.4: Error Handling & UX
- Progress indicators during training
- Error dialogs with helpful messages
- Success notifications
- Disable UI appropriately (prevent changes during training)
- Keyboard shortcuts (optional): Space to start/stop

### Step 5.5: Auto-Parameter Button Implementation
- **"Auto-Suggest Parameters"** button in preset section
- On click:
  1. Analyze dataset (show progress)
  2. Generate suggestions
  3. Populate all parameter fields
  4. Show info message: "Parameters suggested based on [X] labeled tiles, [Y] classes"
  5. User can still modify before training

---

## Implementation Order

### Week 1: Backend Foundation
1. **Step 1.1**: Update dependencies (PyTorch, PyQtGraph)
2. **Step 1.2**: Dataset loading module
3. **Step 1.3**: Architecture config system
4. **Step 1.4**: Model builder (PyTorch CNN)
5. **Step 1.5**: Parameter suggester
6. **Step 1.6**: Training run management
7. **Step 1.7**: Preset management

### Week 2: UI Parameters
8. **Step 2.1**: Basic layout structure
9. **Step 2.2**: Preset management UI
10. **Step 2.3**: Model architecture parameters UI
11. **Step 2.4**: Training hyperparameters UI
12. **Step 2.5**: Advanced options UI
13. **Step 2.6**: Data selection UI

### Week 3: Execution & Monitoring
14. **Step 4.1**: Training thread
15. **Step 4.2**: Run button & controls
16. **Step 4.3**: Parameter validation
17. **Step 4.4**: GPU detection
18. **Step 3.1**: PyQtGraph integration
19. **Step 3.2**: Graph layout
20. **Step 3.3**: Metrics collection

### Week 4: Polish & Integration
21. **Step 5.1**: Project integration
22. **Step 5.2**: Checkpoint management UI
23. **Step 5.3**: Export & results
24. **Step 5.4**: Error handling & UX
25. **Step 5.5**: Auto-parameter button

---

## File Structure After Implementation

```
src/ofc/
├── core/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py          # ArchitectureConfig, TrainingConfig
│   │   ├── pytorch_cnn.py     # ConfigurableCNN, PytorchTrainer
│   │   ├── auto_params.py     # ParameterSuggester
│   │   ├── presets.py         # PresetManager
│   │   ├── device.py          # GPU detection utilities
│   │   └── base.py            # (kept for compatibility, may be empty)
│   ├── dataset.py             # TrainingDataset, data loading
│   └── runs.py                # TrainingRun, TrainingHistory
├── gui/
│   ├── tabs/
│   │   └── train_tab.py       # Complete training tab UI
│   └── widgets/
│       └── training_plot.py   # PyQtGraph plotting widget
```

---

## Testing Strategy

1. **Unit Tests**:
   - Dataset loading with mock data
   - Architecture config validation
   - Parameter suggester heuristics
   - Model building from configs

2. **Integration Tests**:
   - Full training loop (small dataset)
   - Checkpoint save/load
   - Preset save/load
   - GPU/CPU device switching

3. **Manual Testing**:
   - UI responsiveness during training
   - Graph updates in real-time
   - Parameter validation
   - Error handling

---

## Notes & Considerations

1. **PyTorch Installation**: Users need to install PyTorch with CUDA separately if they want GPU support. We'll detect and handle gracefully.

2. **Memory Management**: Large batch sizes on GPU may cause OOM. Add warnings/auto-adjustment.

3. **Training Interruption**: Graceful stop should save checkpoint.

4. **Config Versioning**: Future-proof config format (version field) for compatibility.

5. **Performance**: Graph updates every epoch (not every batch) to avoid UI lag.

6. **Preset Sharing**: JSON format allows easy sharing between users/projects.

---

## Ready to Start?

We'll begin with **Step 1.1: Update Dependencies** and proceed step-by-step. Each step will be tested before moving to the next.
