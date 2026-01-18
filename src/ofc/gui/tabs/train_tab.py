"""Training tab implementation."""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from ofc.core import (
    DatasetStats,
    GridSpec,
    LabelsStore,
    OceanProject,
    get_dataset_stats,
)
from ofc.core.training import (
    ArchitectureConfig,
    LayerConfig,
    ParameterSuggester,
    PresetManager,
    TrainingConfig,
    detect_gpu,
    is_gpu_available,
    list_available_devices,
)
from ofc.core.training.pytorch_cnn import PytorchTrainer
from ofc.core import (
    TrainingDataset,
    get_train_val_split,
    get_class_weights,
    TrainingHistory,
    TrainingRun,
    create_run,
)


class TrainingThread(QThread):
    """Thread for running training in the background."""
    
    # Signals for real-time updates
    epoch_completed = Signal(int, dict)  # epoch, metrics dict
    training_finished = Signal(TrainingHistory)
    training_error = Signal(str)
    status_update = Signal(str)
    
    def __init__(self, config: TrainingConfig, project: OceanProject, training_run: TrainingRun):
        """Initialize training thread."""
        super().__init__()
        self.config = config
        self.project = project
        self.training_run = training_run
        self._stop_requested = False
        
    def stop(self):
        """Request training to stop gracefully."""
        self._stop_requested = True
        
    def run(self):
        """Run the training loop."""
        import time
        from datetime import datetime
        
        def log_progress(message: str):
            """Log progress with timestamp."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
        
        try:
            from torch.utils.data import DataLoader
            
            log_progress("=" * 60)
            log_progress("TRAINING STARTED")
            log_progress("=" * 60)
            start_time = time.time()
            
            self.status_update.emit("Loading dataset...")
            log_progress("Step 1: Loading dataset...")
            dataset_load_start = time.time()
            
            # Load dataset
            grid = self.project.get_grid()
            labels = self.project.get_labels()
            
            # Get classes from labels (actual classes used) rather than classes.json
            # This ensures we use the classes that are actually in the data
            classes_in_labels = set()
            for _, _, _, label in labels.iter_rows():
                if label and label.strip():
                    classes_in_labels.add(label.strip())
            
            if not classes_in_labels:
                log_progress("ERROR: No labeled data available")
                self.training_error.emit("No labeled data available for training")
                return
            
            classes = sorted(list(classes_in_labels))
            log_progress(f"  Found {len(classes)} classes: {classes}")
            
            # Create a cache for source images to speed up data loading
            # Cache up to 64 full images (since multiple tiles come from the same image)
            from ofc.core.io_images import TileCache
            image_cache = TileCache(max_items=64)
            log_progress(f"  Created image cache (max 64 images)")
            
            # Create dataset with cache
            log_progress("  Creating TrainingDataset...")
            dataset = TrainingDataset(self.project, grid, labels, classes=classes, transform=None, cache=image_cache)
            
            if len(dataset) == 0:
                log_progress("ERROR: Dataset is empty")
                self.training_error.emit("No labeled data available for training")
                return
            
            log_progress(f"  Dataset created: {len(dataset)} labeled tiles")
            dataset_load_time = time.time() - dataset_load_start
            log_progress(f"  Dataset loading took {dataset_load_time:.2f} seconds")
            
            # Split into train/val
            log_progress("Step 2: Splitting dataset...")
            split_start = time.time()
            train_dataset, val_dataset = get_train_val_split(
                dataset, 
                val_split=self.config.validation_split,
                seed=self.config.seed
            )
            train_size = len(train_dataset) if train_dataset else 0
            val_size = len(val_dataset) if val_dataset else 0
            log_progress(f"  Train: {train_size} samples, Val: {val_size} samples")
            split_time = time.time() - split_start
            log_progress(f"  Dataset splitting took {split_time:.2f} seconds")
            
            # Create data loaders
            log_progress("Step 3: Creating data loaders...")
            loader_start = time.time()
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,  # 0 for Windows compatibility
                pin_memory=self.config.use_gpu
            )
            
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=self.config.use_gpu
                )
            
            num_train_batches = len(train_loader)
            num_val_batches = len(val_loader) if val_loader else 0
            log_progress(f"  Train batches: {num_train_batches}, Val batches: {num_val_batches}")
            loader_time = time.time() - loader_start
            log_progress(f"  Data loader creation took {loader_time:.2f} seconds")
            
            # Get class weights if requested
            class_weights = None
            if self.config.class_weights:
                log_progress("Step 4: Calculating class weights...")
                weights_start = time.time()
                class_weights = get_class_weights(dataset, classes)
                weights_time = time.time() - weights_start
                log_progress(f"  Class weights calculated in {weights_time:.2f} seconds")
            
            # Create trainer
            log_progress("Step 5: Initializing trainer...")
            trainer_start = time.time()
            self.status_update.emit("Initializing trainer...")
            trainer = PytorchTrainer(self.config, self.project)
            trainer_time = time.time() - trainer_start
            log_progress(f"  Trainer initialized in {trainer_time:.2f} seconds")
            log_progress(f"  Device: {trainer.device}")
            log_progress(f"  Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
            
            # Save config
            log_progress("Step 6: Saving configuration...")
            config_start = time.time()
            self.training_run.save_config(self.config)
            config_time = time.time() - config_start
            log_progress(f"  Config saved in {config_time:.2f} seconds")
            
            # Initialize history
            history = TrainingHistory(
                epochs=[],
                train_loss=[],
                train_accuracy=[],
                val_loss=[],
                val_accuracy=[],
                learning_rates=[],
                timestamps=[]
            )
            
            # Progress callback
            epoch_times = []
            
            def progress_callback(epoch, metrics):
                if self._stop_requested:
                    return
                
                epoch_num = epoch + 1  # 1-indexed
                epoch_time = time.time()
                if epoch_num > 1:
                    prev_epoch_time = epoch_times[-1] if epoch_times else epoch_time
                    elapsed = epoch_time - prev_epoch_time
                    log_progress(f"Epoch {epoch_num-1} completed in {elapsed:.2f} seconds")
                
                epoch_times.append(epoch_time)
                
                # Update history
                history.add_epoch(
                    epoch=epoch_num,
                    train_loss=metrics.get("loss", 0.0),
                    train_accuracy=metrics.get("accuracy", 0.0),
                    val_loss=metrics.get("val_loss"),
                    val_accuracy=metrics.get("val_accuracy"),
                    learning_rate=metrics.get("learning_rate")
                )
                
                # Log epoch metrics
                log_progress(f"Epoch {epoch_num}/{self.config.num_epochs} metrics:")
                log_progress(f"  Train Loss: {metrics.get('loss', 0.0):.4f}, Train Acc: {metrics.get('accuracy', 0.0):.4%}")
                if metrics.get('val_loss') is not None:
                    log_progress(f"  Val Loss: {metrics.get('val_loss'):.4f}, Val Acc: {metrics.get('val_accuracy', 0.0):.4%}")
                
                # Emit signal for UI update
                self.epoch_completed.emit(epoch_num, metrics)
                
                # Save checkpoint periodically
                if epoch_num % self.config.save_checkpoint_every == 0:
                    checkpoint_start = time.time()
                    is_best = metrics.get("val_loss") is not None and metrics.get("val_loss", float('inf')) < trainer.best_val_loss
                    self.training_run.save_checkpoint(
                        trainer.model,
                        epoch_num,
                        metrics,
                        is_best=is_best
                    )
                    checkpoint_time = time.time() - checkpoint_start
                    log_progress(f"  Checkpoint saved in {checkpoint_time:.2f} seconds")
            
            # Start training
            log_progress("=" * 60)
            log_progress("STARTING TRAINING LOOP")
            log_progress(f"  Total epochs: {self.config.num_epochs}")
            log_progress(f"  Batch size: {self.config.batch_size}")
            log_progress(f"  Learning rate: {self.config.learning_rate}")
            log_progress("=" * 60)
            self.status_update.emit("Training started...")
            
            training_start = time.time()
            trainer.train(train_loader, val_loader, progress_callback=progress_callback)
            training_time = time.time() - training_start
            
            log_progress("=" * 60)
            log_progress("TRAINING LOOP COMPLETED")
            log_progress(f"  Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
            if epoch_times:
                avg_epoch_time = sum(epoch_times[i] - epoch_times[i-1] for i in range(1, len(epoch_times))) / (len(epoch_times) - 1) if len(epoch_times) > 1 else 0
                log_progress(f"  Average epoch time: {avg_epoch_time:.2f} seconds")
            log_progress("=" * 60)
            
            # Save final history
            log_progress("Saving final history...")
            history_start = time.time()
            self.training_run.save_history(history)
            history_time = time.time() - history_start
            log_progress(f"  History saved in {history_time:.2f} seconds")
            
            # Save final checkpoint
            log_progress("Saving final checkpoint...")
            final_checkpoint_start = time.time()
            final_metrics = {"loss": history.train_loss[-1] if history.train_loss else 0.0}
            self.training_run.save_checkpoint(
                trainer.model,
                len(history.epochs),
                final_metrics,
                is_best=False
            )
            final_checkpoint_time = time.time() - final_checkpoint_start
            log_progress(f"  Final checkpoint saved in {final_checkpoint_time:.2f} seconds")
            
            total_time = time.time() - start_time
            log_progress("=" * 60)
            log_progress("TRAINING COMPLETED")
            log_progress(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            log_progress("=" * 60)
            
            if not self._stop_requested:
                self.status_update.emit("Training completed!")
                self.training_finished.emit(history)
            else:
                self.status_update.emit("Training stopped by user")
                log_progress("Training was stopped by user")
                
        except Exception as e:
            import traceback
            error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
            log_progress(f"ERROR: {error_msg}")
            self.training_error.emit(error_msg)


class TrainTab(QWidget):
    """Training tab for model training."""

    def __init__(self):
        """Initialize training tab."""
        super().__init__()
        self.project: Optional[OceanProject] = None
        self.grid: Optional[GridSpec] = None
        self.labels: Optional[LabelsStore] = None
        self.preset_manager: Optional[PresetManager] = None
        self.current_config: Optional[TrainingConfig] = None
        self.layers_data: list[dict] = []  # Store layer configs for UI
        self.training_thread: Optional[TrainingThread] = None
        self.current_run: Optional[TrainingRun] = None

        self.init_ui()

    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Main splitter: Parameters (top) and Graphs/Controls (bottom)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(main_splitter)

        # ===== TOP SECTION: Parameters (Scrollable) =====
        parameters_section = self.create_parameters_section()
        main_splitter.addWidget(parameters_section)

        # ===== BOTTOM SECTION: Graphs and Controls =====
        bottom_section = self.create_bottom_section()
        main_splitter.addWidget(bottom_section)

        # Set splitter proportions (parameters get more space initially)
        main_splitter.setSizes([600, 400])

    def create_parameters_section(self) -> QWidget:
        """Create the parameters section with scrollable content."""
        # Container widget
        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Content widget (will contain all parameter groups)
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)

        # Dataset Info Group Box (first - at the top)
        dataset_info_group = self.create_dataset_info_group()
        content_layout.addWidget(dataset_info_group)

        # Model Architecture and Training Hyperparameters side-by-side
        arch_hyperparams_container = QWidget()
        arch_hyperparams_layout = QHBoxLayout()
        arch_hyperparams_container.setLayout(arch_hyperparams_layout)

        # Model Architecture Group Box (left)
        architecture_group = self.create_architecture_group()
        arch_hyperparams_layout.addWidget(architecture_group)

        # Training Hyperparameters Group Box (right)
        hyperparams_group = self.create_hyperparameters_group()
        arch_hyperparams_layout.addWidget(hyperparams_group)

        content_layout.addWidget(arch_hyperparams_container)

        content_layout.addStretch()

        scroll.setWidget(content_widget)
        container_layout.addWidget(scroll)

        return container

    def create_bottom_section(self) -> QWidget:
        """Create the bottom section with graphs and controls."""
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        # Splitter for graphs (left) and controls (right)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Graphs section (left)
        graphs_section = self.create_graphs_section()
        bottom_splitter.addWidget(graphs_section)

        # Controls section (right) - will be small
        controls_section = self.create_controls_section()
        bottom_splitter.addWidget(controls_section)

        # Set proportions (graphs get most space)
        bottom_splitter.setSizes([800, 200])

        layout.addWidget(bottom_splitter)

        return container

    def create_graphs_section(self) -> QWidget:
        """Create the graphs section for training metrics."""
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        # Title
        title = QLabel("Training Metrics")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Training plot widget (PyQtGraph)
        from ofc.gui.widgets.training_plot import TrainingPlotWidget
        
        self.training_plot = TrainingPlotWidget()
        self.training_plot.setMinimumHeight(300)
        layout.addWidget(self.training_plot)

        return container

    def create_controls_section(self) -> QWidget:
        """Create the controls section with buttons and status."""
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        # Title
        title = QLabel("Training Controls")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Control buttons
        self.start_button = QPushButton("Start Training")
        self.start_button.setEnabled(False)  # Disabled until project is set
        self.start_button.clicked.connect(self.on_start_training)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.on_stop_training)
        layout.addWidget(self.stop_button)

        self.save_model_button = QPushButton("Save Model")
        self.save_model_button.setEnabled(False)
        self.save_model_button.clicked.connect(self.on_save_model)
        layout.addWidget(self.save_model_button)

        layout.addStretch()

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        return container

    def create_dataset_info_group(self) -> QGroupBox:
        """Create the dataset information group box."""
        group = QGroupBox("Dataset Information")
        layout = QHBoxLayout()  # Changed to horizontal for side-by-side layout
        group.setLayout(layout)
        
        # Left side: Dataset stats
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_widget.setLayout(stats_layout)

        # Create labels for dataset stats
        self.dataset_info_labels = {}
        
        # Raw images count
        raw_images_row = QHBoxLayout()
        raw_images_row.addWidget(QLabel("Raw Images:"))
        self.dataset_info_labels["raw_images"] = QLabel("N/A")
        self.dataset_info_labels["raw_images"].setStyleSheet("font-weight: bold;")
        raw_images_row.addWidget(self.dataset_info_labels["raw_images"])
        raw_images_row.addStretch()
        stats_layout.addLayout(raw_images_row)
        
        # Labeled tiles count
        labeled_tiles_row = QHBoxLayout()
        labeled_tiles_row.addWidget(QLabel("Labeled Tiles:"))
        self.dataset_info_labels["labeled_tiles"] = QLabel("N/A")
        self.dataset_info_labels["labeled_tiles"].setStyleSheet("font-weight: bold;")
        labeled_tiles_row.addWidget(self.dataset_info_labels["labeled_tiles"])
        labeled_tiles_row.addStretch()
        stats_layout.addLayout(labeled_tiles_row)
        
        # Images with labels count
        images_with_labels_row = QHBoxLayout()
        images_with_labels_row.addWidget(QLabel("Images with Labels:"))
        self.dataset_info_labels["images_with_labels"] = QLabel("N/A")
        self.dataset_info_labels["images_with_labels"].setStyleSheet("font-weight: bold;")
        images_with_labels_row.addWidget(self.dataset_info_labels["images_with_labels"])
        images_with_labels_row.addStretch()
        stats_layout.addLayout(images_with_labels_row)
        
        # Number of classes
        num_classes_row = QHBoxLayout()
        num_classes_row.addWidget(QLabel("Number of Classes:"))
        self.dataset_info_labels["num_classes"] = QLabel("N/A")
        self.dataset_info_labels["num_classes"].setStyleSheet("font-weight: bold;")
        num_classes_row.addWidget(self.dataset_info_labels["num_classes"])
        num_classes_row.addStretch()
        stats_layout.addLayout(num_classes_row)
        
        # Average tiles per image
        avg_tiles_row = QHBoxLayout()
        avg_tiles_row.addWidget(QLabel("Average Tiles per Image:"))
        self.dataset_info_labels["avg_tiles"] = QLabel("N/A")
        self.dataset_info_labels["avg_tiles"].setStyleSheet("font-weight: bold;")
        avg_tiles_row.addWidget(self.dataset_info_labels["avg_tiles"])
        avg_tiles_row.addStretch()
        stats_layout.addLayout(avg_tiles_row)
        
        # Input size row (read-only, from grid metadata)
        input_size_row = QHBoxLayout()
        input_size_row.addWidget(QLabel("Input Size:"))
        
        self.dataset_info_labels["input_size"] = QLabel("N/A")
        self.dataset_info_labels["input_size"].setStyleSheet("font-weight: bold;")
        input_size_row.addWidget(self.dataset_info_labels["input_size"])
        
        input_size_row.addStretch()
        stats_layout.addLayout(input_size_row)
        
        # Class distribution (if available)
        self.class_distribution_label = QLabel("")
        self.class_distribution_label.setWordWrap(True)
        self.class_distribution_label.setStyleSheet("color: #666; font-size: 10px;")
        stats_layout.addWidget(self.class_distribution_label)
        
        # Add stats widget to left side
        layout.addWidget(stats_widget)
        
        # Middle: Class distribution histogram
        histogram_widget = QWidget()
        histogram_layout = QVBoxLayout()
        histogram_widget.setLayout(histogram_layout)
        histogram_layout.addWidget(QLabel("Class Distribution:"))
        
        # Create PyQtGraph histogram
        self.class_distribution_plot = pg.PlotWidget()
        self.class_distribution_plot.setMinimumHeight(150)
        self.class_distribution_plot.setMaximumHeight(200)
        self.class_distribution_plot.setLabel('left', 'Count')
        self.class_distribution_plot.setLabel('bottom', 'Class')
        self.class_distribution_plot.showGrid(x=True, y=True, alpha=0.3)
        histogram_layout.addWidget(self.class_distribution_plot)
        
        layout.addWidget(histogram_widget)
        
        # Right side: Data augmentation block (moved from Data Settings)
        aug_group = self.create_data_augmentation_group()
        layout.addWidget(aug_group)

        return group

    def update_dataset_info(self):
        """Update the dataset information display."""
        if not self.project or not self.grid or not self.labels:
            # Reset to N/A
            for label in self.dataset_info_labels.values():
                label.setText("N/A")
            self.class_distribution_label.setText("")
            return

        try:
            # Get raw images count
            raw_images = self.project.list_raw_images()
            num_raw_images = len(raw_images)
            self.dataset_info_labels["raw_images"].setText(str(num_raw_images))

            # Get dataset statistics
            stats: DatasetStats = get_dataset_stats(self.project, self.grid, self.labels)
            
            self.dataset_info_labels["labeled_tiles"].setText(str(stats.total_labeled_tiles))
            self.dataset_info_labels["images_with_labels"].setText(str(stats.images_with_labels))
            self.dataset_info_labels["num_classes"].setText(str(stats.num_classes))
            self.dataset_info_labels["avg_tiles"].setText(f"{stats.average_tiles_per_image:.1f}")
            
            # Update input size from grid
            if self.grid:
                input_size_text = f"{self.grid.tile_w} × {self.grid.tile_h}"
                self.dataset_info_labels["input_size"].setText(input_size_text)
            else:
                self.dataset_info_labels["input_size"].setText("N/A")

            # Format class distribution
            if stats.class_counts:
                dist_text = "Class Distribution: "
                dist_parts = []
                for class_name, count in sorted(stats.class_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = stats.class_distribution.get(class_name, 0.0)
                    dist_parts.append(f"{class_name}: {count} ({percentage:.1f}%)")
                dist_text += " | ".join(dist_parts)
                self.class_distribution_label.setText(dist_text)
                
                # Update histogram
                self.update_class_distribution_histogram(stats)
            else:
                self.class_distribution_label.setText("No labeled data available")
                if hasattr(self, 'class_distribution_plot'):
                    self.class_distribution_plot.clear()

        except Exception as e:
            # On error, show error message
            for label in self.dataset_info_labels.values():
                label.setText("Error")
            self.class_distribution_label.setText(f"Error loading dataset info: {str(e)}")
            self.class_distribution_plot.clear()
    
    def update_class_distribution_histogram(self, stats: DatasetStats):
        """Update the class distribution histogram."""
        if not hasattr(self, 'class_distribution_plot'):
            return
        
        self.class_distribution_plot.clear()
        
        if not stats.class_counts:
            return
        
        # Sort classes by count for better visualization
        sorted_classes = sorted(stats.class_counts.items(), key=lambda x: x[1], reverse=True)
        class_names = [name for name, _ in sorted_classes]
        counts = [count for _, count in sorted_classes]
        
        # Create bar chart
        x_positions = list(range(len(class_names)))
        bg = pg.BarGraphItem(x=x_positions, height=counts, width=0.6, brush='b')
        self.class_distribution_plot.addItem(bg)
        
        # Set x-axis labels
        axis = self.class_distribution_plot.getAxis('bottom')
        axis.setTicks([[(i, name) for i, name in enumerate(class_names)]])
        
        # Update y-axis range
        if counts:
            max_count = max(counts)
            self.class_distribution_plot.setYRange(0, max_count * 1.1)

    def create_preset_management_group(self) -> QWidget:
        """Create the preset management widget (can be embedded in other groups)."""
        container = QWidget()
        layout = QHBoxLayout()
        container.setLayout(layout)

        # All preset controls on one row
        layout.addWidget(QLabel("Preset:"))
        
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(False)
        self.preset_combo.currentTextChanged.connect(self.on_preset_selected)
        layout.addWidget(self.preset_combo)
        
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.on_load_preset)
        layout.addWidget(load_btn)
        
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self.on_export_config)
        layout.addWidget(export_btn)
        
        import_btn = QPushButton("Import...")
        import_btn.clicked.connect(self.on_import_config)
        layout.addWidget(import_btn)
        
        auto_suggest_btn = QPushButton("Auto-Suggest Parameters")
        auto_suggest_btn.clicked.connect(self.on_auto_suggest)
        layout.addWidget(auto_suggest_btn)
        
        layout.addStretch()
        
        return container

    def create_architecture_group(self) -> QGroupBox:
        """Create the model architecture configuration group box."""
        group = QGroupBox("Model Architecture")
        layout = QVBoxLayout()
        group.setLayout(layout)

        # Preset Management (moved inside architecture group)
        preset_group = self.create_preset_management_group()
        layout.addWidget(preset_group)

        # Layers table
        layers_label = QLabel("Layers:")
        layout.addWidget(layers_label)

        # Table for layers
        self.layers_table = QTableWidget()
        self.layers_table.setColumnCount(6)  # Added checkbox column
        self.layers_table.setHorizontalHeaderLabels(
            ["", "Type", "Parameters", "Activation", "", ""]
        )
        self.layers_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.layers_table.setMinimumHeight(200)
        
        # Set column widths to fit content
        # Checkbox column - small fixed width
        self.layers_table.setColumnWidth(0, 30)
        # Type column - fit longest layer type name
        self.layers_table.setColumnWidth(1, 120)
        # Parameters column - needs most space for multiple widgets
        self.layers_table.setColumnWidth(2, 400)
        # Activation column - fit longest activation name
        self.layers_table.setColumnWidth(3, 100)
        # Up button column - small fixed width
        self.layers_table.setColumnWidth(4, 40)
        # Down button column - small fixed width
        self.layers_table.setColumnWidth(5, 40)
        
        # Make Parameters column stretchable, others fixed
        header = self.layers_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Checkbox
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)  # Type
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Parameters - stretches
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)  # Activation
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)  # Up button
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)  # Down button
        
        layout.addWidget(self.layers_table)

        # Layer controls
        layer_controls = QHBoxLayout()
        
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.on_add_layer)
        layer_controls.addWidget(add_layer_btn)
        
        remove_layer_btn = QPushButton("Remove Selected")
        remove_layer_btn.clicked.connect(self.on_remove_layer)
        layer_controls.addWidget(remove_layer_btn)
        
        move_up_btn = QPushButton("Move Up")
        move_up_btn.clicked.connect(self.on_move_layer_up)
        layer_controls.addWidget(move_up_btn)
        
        move_down_btn = QPushButton("Move Down")
        move_down_btn.clicked.connect(self.on_move_layer_down)
        layer_controls.addWidget(move_down_btn)
        
        layer_controls.addStretch()
        layout.addLayout(layer_controls)

        # Number of classes
        classes_row = QHBoxLayout()
        classes_row.addWidget(QLabel("Number of Classes:"))
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setMinimum(1)
        self.num_classes_spin.setMaximum(1000)
        self.num_classes_spin.setValue(10)
        classes_row.addWidget(self.num_classes_spin)
        classes_row.addStretch()
        layout.addLayout(classes_row)

        # Connect table selection to show layer editor
        self.layers_table.cellClicked.connect(self.on_layer_selected)

        return group

    def on_add_layer(self):
        """Add a new layer to the architecture."""
        row = self.layers_table.rowCount()
        self.layers_table.insertRow(row)
        
        # Checkbox for selection
        checkbox = QCheckBox()
        checkbox.setChecked(False)
        self.layers_table.setCellWidget(row, 0, checkbox)
        
        # Layer type combo
        type_combo = QComboBox()
        type_combo.addItems(["conv2d", "maxpool2d", "avgpool2d", "linear", "dropout", "batchnorm2d"])
        type_combo.currentTextChanged.connect(lambda: self.on_layer_type_changed(row))
        self.layers_table.setCellWidget(row, 1, type_combo)
        
        # Parameters (will be populated based on type)
        params_widget = QWidget()
        params_layout = QHBoxLayout()
        params_widget.setLayout(params_layout)
        params_layout.setContentsMargins(2, 2, 2, 2)
        self.layers_table.setCellWidget(row, 2, params_widget)
        
        # Activation combo
        activation_combo = QComboBox()
        activation_combo.addItems(["None", "relu", "sigmoid", "tanh", "softmax"])
        self.layers_table.setCellWidget(row, 3, activation_combo)
        
        # Up/Down buttons
        up_btn = QPushButton("↑")
        up_btn.setMaximumWidth(30)
        up_btn.clicked.connect(lambda checked, r=row: self.on_move_layer_up_at_row(r))
        self.layers_table.setCellWidget(row, 4, up_btn)
        
        down_btn = QPushButton("↓")
        down_btn.setMaximumWidth(30)
        down_btn.clicked.connect(lambda checked, r=row: self.on_move_layer_down_at_row(r))
        self.layers_table.setCellWidget(row, 5, down_btn)
        
        # Initialize layer data
        self.layers_data.append({
            "layer_type": "conv2d",
            "params": {},
            "activation": None,
        })
        
        # Populate parameters for default type
        self.update_layer_params_widget(row, "conv2d")

    def on_remove_layer(self):
        """Remove selected layer(s) based on checkboxes."""
        selected_rows = set()
        
        # Check both checkboxes and table selection
        for row in range(self.layers_table.rowCount()):
            checkbox = self.layers_table.cellWidget(row, 0)
            if checkbox and isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                selected_rows.add(row)
        
        # Also include table-selected rows if no checkboxes are checked
        if not selected_rows:
            for item in self.layers_table.selectedItems():
                selected_rows.add(item.row())
        
        # Remove in reverse order to maintain indices
        for row in sorted(selected_rows, reverse=True):
            self.layers_table.removeRow(row)
            if row < len(self.layers_data):
                self.layers_data.pop(row)

    def on_move_layer_up(self):
        """Move selected layer up."""
        current_row = self.layers_table.currentRow()
        if current_row > 0:
            self.on_move_layer_up_at_row(current_row)

    def on_move_layer_up_at_row(self, row: int):
        """Move layer at specific row up."""
        if row <= 0 or row >= len(self.layers_data):
            return
        
        # Swap data first
        if row < len(self.layers_data) and row - 1 < len(self.layers_data):
            self.layers_data[row], self.layers_data[row - 1] = (
                self.layers_data[row - 1],
                self.layers_data[row],
            )
        
        # Swap rows in table
        self.swap_table_rows(row, row - 1)
        
        # Select the moved row
        self.layers_table.selectRow(row - 1)

    def on_move_layer_down(self):
        """Move selected layer down."""
        current_row = self.layers_table.currentRow()
        if current_row >= 0 and current_row < self.layers_table.rowCount() - 1:
            self.on_move_layer_down_at_row(current_row)

    def on_move_layer_down_at_row(self, row: int):
        """Move layer at specific row down."""
        if row < 0 or row >= self.layers_table.rowCount() - 1 or row >= len(self.layers_data) - 1:
            return
        
        # Swap data first
        if row < len(self.layers_data) and row + 1 < len(self.layers_data):
            self.layers_data[row], self.layers_data[row + 1] = (
                self.layers_data[row + 1],
                self.layers_data[row],
            )
        
        # Swap rows in table
        self.swap_table_rows(row, row + 1)
        
        # Select the moved row
        self.layers_table.selectRow(row + 1)

    def swap_table_rows(self, row1: int, row2: int):
        """Swap two rows in the table by extracting data and rebuilding."""
        # Extract all data from both rows
        row1_data = self._extract_row_data(row1)
        row2_data = self._extract_row_data(row2)
        
        # Remove both rows
        self.layers_table.removeRow(row2)  # Remove higher row first
        self.layers_table.removeRow(row1)
        
        # Insert rows back in swapped order
        self.layers_table.insertRow(row1)
        self.layers_table.insertRow(row2)
        
        # Rebuild row1 with row2's data
        self._rebuild_row(row1, row2_data)
        # Rebuild row2 with row1's data
        self._rebuild_row(row2, row1_data)
    
    def _extract_row_data(self, row: int) -> dict:
        """Extract all data from a table row."""
        data = {
            "checkbox_checked": False,
            "layer_type": "conv2d",
            "params": {},
            "activation": None,
            "params_widget_data": {},
        }
        
        # Extract checkbox state
        checkbox = self.layers_table.cellWidget(row, 0)
        if checkbox and isinstance(checkbox, QCheckBox):
            data["checkbox_checked"] = checkbox.isChecked()
        
        # Extract layer type
        type_combo = self.layers_table.cellWidget(row, 1)
        if type_combo and isinstance(type_combo, QComboBox):
            data["layer_type"] = type_combo.currentText()
        
        # Extract parameters widget data
        params_widget = self.layers_table.cellWidget(row, 2)
        if params_widget:
            data["params"] = self.extract_layer_params(params_widget, data["layer_type"])
        
        # Extract activation
        activation_combo = self.layers_table.cellWidget(row, 3)
        if activation_combo and isinstance(activation_combo, QComboBox):
            activation_text = activation_combo.currentText()
            data["activation"] = None if activation_text == "None" else activation_text
        
        return data
    
    def _rebuild_row(self, row: int, data: dict):
        """Rebuild a table row with the given data."""
        # Checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(data.get("checkbox_checked", False))
        self.layers_table.setCellWidget(row, 0, checkbox)
        
        # Layer type combo
        type_combo = QComboBox()
        type_combo.addItems(["conv2d", "maxpool2d", "avgpool2d", "linear", "dropout", "batchnorm2d"])
        type_combo.setCurrentText(data.get("layer_type", "conv2d"))
        type_combo.currentTextChanged.connect(lambda: self.on_layer_type_changed(row))
        self.layers_table.setCellWidget(row, 1, type_combo)
        
        # Parameters widget
        layer_type = data.get("layer_type", "conv2d")
        self.update_layer_params_widget(row, layer_type)
        self.set_layer_params_values(row, data.get("params", {}), layer_type)
        
        # Activation combo
        activation_combo = QComboBox()
        activation_combo.addItems(["None", "relu", "sigmoid", "tanh", "softmax"])
        activation_text = data.get("activation")
        activation_combo.setCurrentText(activation_text if activation_text else "None")
        self.layers_table.setCellWidget(row, 3, activation_combo)
        
        # Up/Down buttons
        up_btn = QPushButton("↑")
        up_btn.setMaximumWidth(30)
        up_btn.clicked.connect(lambda checked, r=row: self.on_move_layer_up_at_row(r))
        self.layers_table.setCellWidget(row, 4, up_btn)
        
        down_btn = QPushButton("↓")
        down_btn.setMaximumWidth(30)
        down_btn.clicked.connect(lambda checked, r=row: self.on_move_layer_down_at_row(r))
        self.layers_table.setCellWidget(row, 5, down_btn)

    def on_layer_type_changed(self, row: int):
        """Handle layer type change - update parameters widget."""
        type_combo = self.layers_table.cellWidget(row, 1)  # Column 1 is now Type
        if type_combo:
            layer_type = type_combo.currentText()
            self.update_layer_params_widget(row, layer_type)
            if row < len(self.layers_data):
                self.layers_data[row]["layer_type"] = layer_type

    def update_layer_params_widget(self, row: int, layer_type: str):
        """Update the parameters widget for a layer based on its type."""
        params_widget = QWidget()
        params_layout = QHBoxLayout()
        params_widget.setLayout(params_layout)
        params_layout.setContentsMargins(2, 2, 2, 2)
        
        if layer_type == "conv2d":
            params_layout.addWidget(QLabel("Out Channels:"))
            out_channels = QSpinBox()
            out_channels.setMinimum(1)
            out_channels.setMaximum(1024)
            out_channels.setValue(32)
            params_layout.addWidget(out_channels)
            
            params_layout.addWidget(QLabel("Kernel:"))
            kernel = QSpinBox()
            kernel.setMinimum(1)
            kernel.setMaximum(15)
            kernel.setValue(3)
            kernel.setSingleStep(2)
            params_layout.addWidget(kernel)
            
            params_layout.addWidget(QLabel("Stride:"))
            stride = QSpinBox()
            stride.setMinimum(1)
            stride.setMaximum(10)
            stride.setValue(1)
            params_layout.addWidget(stride)
            
            params_layout.addWidget(QLabel("Padding:"))
            padding = QSpinBox()
            padding.setMinimum(0)
            padding.setMaximum(10)
            padding.setValue(1)
            params_layout.addWidget(padding)
            
        elif layer_type == "linear":
            params_layout.addWidget(QLabel("Out Features:"))
            out_features = QSpinBox()
            out_features.setMinimum(1)
            out_features.setMaximum(10000)
            out_features.setValue(128)
            params_layout.addWidget(out_features)
            
        elif layer_type == "dropout":
            params_layout.addWidget(QLabel("Probability:"))
            prob = QDoubleSpinBox()
            prob.setMinimum(0.0)
            prob.setMaximum(1.0)
            prob.setValue(0.5)
            prob.setSingleStep(0.1)
            prob.setDecimals(2)
            params_layout.addWidget(prob)
            
        elif layer_type in ("maxpool2d", "avgpool2d"):
            params_layout.addWidget(QLabel("Kernel:"))
            kernel = QSpinBox()
            kernel.setMinimum(2)
            kernel.setMaximum(10)
            kernel.setValue(2)
            kernel.setSingleStep(2)
            params_layout.addWidget(kernel)
            
            params_layout.addWidget(QLabel("Stride:"))
            stride = QSpinBox()
            stride.setMinimum(1)
            stride.setMaximum(10)
            stride.setValue(2)
            params_layout.addWidget(stride)
            
        # batchnorm2d has no parameters
        
        params_layout.addStretch()
        self.layers_table.setCellWidget(row, 2, params_widget)  # Column 2 is now Parameters

    def on_layer_selected(self, row: int, col: int):
        """Handle layer selection (for future layer editor panel)."""
        # Could show detailed editor in a side panel
        pass

    def get_architecture_config(self) -> Optional[ArchitectureConfig]:
        """
        Get ArchitectureConfig from current UI state.

        Returns:
            ArchitectureConfig or None if invalid
        """
        try:
            # Get input size from grid (read-only)
            if self.grid:
                input_size = (self.grid.tile_w, self.grid.tile_h)
            else:
                input_size = (256, 256)  # Default fallback
            num_classes = self.num_classes_spin.value()
            
            layers = []
            for i in range(self.layers_table.rowCount()):
                type_combo = self.layers_table.cellWidget(i, 1)  # Column 1 is Type
                activation_combo = self.layers_table.cellWidget(i, 3)  # Column 3 is Activation
                params_widget = self.layers_table.cellWidget(i, 2)  # Column 2 is Parameters
                
                if not type_combo:
                    continue
                
                layer_type = type_combo.currentText()
                activation = activation_combo.currentText() if activation_combo else None
                if activation == "None":
                    activation = None
                
                # Extract parameters from widget
                params = self.extract_layer_params(params_widget, layer_type)
                
                layers.append(LayerConfig(
                    layer_type=layer_type,
                    params=params,
                    activation=activation,
                ))
            
            if not layers:
                return None
            
            return ArchitectureConfig(
                input_size=input_size,
                layers=layers,
                num_classes=num_classes,
            )
        except Exception as e:
            print(f"Error creating architecture config: {e}")
            return None

    def extract_layer_params(self, params_widget: QWidget, layer_type: str) -> dict:
        """Extract parameters from a layer's params widget."""
        params = {}
        layout = params_widget.layout()
        
        if not layout:
            return params
        
        if layer_type == "conv2d":
            # Find spinboxes in order: out_channels, kernel, stride, padding
            spinboxes = [w for w in self.get_widgets_from_layout(layout) if isinstance(w, QSpinBox)]
            if len(spinboxes) >= 4:
                params["out_channels"] = spinboxes[0].value()
                params["kernel_size"] = spinboxes[1].value()
                params["stride"] = spinboxes[2].value()
                params["padding"] = spinboxes[3].value()
                
        elif layer_type == "linear":
            spinboxes = [w for w in self.get_widgets_from_layout(layout) if isinstance(w, QSpinBox)]
            if len(spinboxes) >= 1:
                params["out_features"] = spinboxes[0].value()
                
        elif layer_type == "dropout":
            doublespinboxes = [w for w in self.get_widgets_from_layout(layout) if isinstance(w, QDoubleSpinBox)]
            if len(doublespinboxes) >= 1:
                params["p"] = doublespinboxes[0].value()
                
        elif layer_type in ("maxpool2d", "avgpool2d"):
            spinboxes = [w for w in self.get_widgets_from_layout(layout) if isinstance(w, QSpinBox)]
            if len(spinboxes) >= 2:
                params["kernel_size"] = spinboxes[0].value()
                params["stride"] = spinboxes[1].value()
        
        return params

    def get_widgets_from_layout(self, layout) -> list:
        """Get all widgets from a layout recursively."""
        widgets = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                widgets.append(item.widget())
            elif item.layout():
                widgets.extend(self.get_widgets_from_layout(item.layout()))
        return widgets

    def set_architecture_config(self, config: ArchitectureConfig):
        """Populate UI with architecture configuration."""
        # Input size is read-only from grid, so we don't set it here
        # The display will be updated via update_dataset_info()
        self.num_classes_spin.setValue(config.num_classes)
        
        # Clear existing layers
        self.layers_table.setRowCount(0)
        self.layers_data.clear()
        
        # Add layers
        for layer_cfg in config.layers:
            row = self.layers_table.rowCount()
            self.layers_table.insertRow(row)
            
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox.setChecked(False)
            self.layers_table.setCellWidget(row, 0, checkbox)
            
            # Layer type
            type_combo = QComboBox()
            type_combo.addItems(["conv2d", "maxpool2d", "avgpool2d", "linear", "dropout", "batchnorm2d"])
            type_combo.setCurrentText(layer_cfg.layer_type)
            type_combo.currentTextChanged.connect(lambda r=row: self.on_layer_type_changed(r))
            self.layers_table.setCellWidget(row, 1, type_combo)  # Column 1 is Type
            
            # Parameters
            self.update_layer_params_widget(row, layer_cfg.layer_type)
            # Set parameter values
            self.set_layer_params_values(row, layer_cfg.params, layer_cfg.layer_type)
            
            # Activation
            activation_combo = QComboBox()
            activation_combo.addItems(["None", "relu", "sigmoid", "tanh", "softmax"])
            activation_text = layer_cfg.activation if layer_cfg.activation else "None"
            activation_combo.setCurrentText(activation_text)
            self.layers_table.setCellWidget(row, 3, activation_combo)  # Column 3 is Activation
            
            # Up/Down buttons
            up_btn = QPushButton("↑")
            up_btn.setMaximumWidth(30)
            up_btn.clicked.connect(lambda checked, r=row: self.on_move_layer_up_at_row(r))
            self.layers_table.setCellWidget(row, 4, up_btn)  # Column 4 is Up button
            
            down_btn = QPushButton("↓")
            down_btn.setMaximumWidth(30)
            down_btn.clicked.connect(lambda checked, r=row: self.on_move_layer_down_at_row(r))
            self.layers_table.setCellWidget(row, 5, down_btn)  # Column 5 is Down button
            
            # Store data
            self.layers_data.append({
                "layer_type": layer_cfg.layer_type,
                "params": layer_cfg.params.copy(),
                "activation": layer_cfg.activation,
            })

    def set_layer_params_values(self, row: int, params: dict, layer_type: str):
        """Set parameter values in a layer's params widget."""
        params_widget = self.layers_table.cellWidget(row, 2)  # Column 2 is Parameters
        if not params_widget:
            return
        
        layout = params_widget.layout()
        if not layout:
            return
        
        widgets = self.get_widgets_from_layout(layout)
        spinboxes = [w for w in widgets if isinstance(w, (QSpinBox, QDoubleSpinBox))]
        
        if layer_type == "conv2d":
            if len(spinboxes) >= 4:
                spinboxes[0].setValue(params.get("out_channels", 32))
                spinboxes[1].setValue(params.get("kernel_size", 3))
                spinboxes[2].setValue(params.get("stride", 1))
                spinboxes[3].setValue(params.get("padding", 1))
        elif layer_type == "linear":
            if len(spinboxes) >= 1:
                spinboxes[0].setValue(params.get("out_features", 128))
        elif layer_type == "dropout":
            if len(spinboxes) >= 1 and isinstance(spinboxes[0], QDoubleSpinBox):
                spinboxes[0].setValue(params.get("p", 0.5))
        elif layer_type in ("maxpool2d", "avgpool2d"):
            if len(spinboxes) >= 2:
                spinboxes[0].setValue(params.get("kernel_size", 2))
                spinboxes[1].setValue(params.get("stride", 2))

    def set_project(
        self, project: OceanProject, grid: GridSpec, labels: LabelsStore
    ):
        """
        Set the current project and update UI.

        Args:
            project: OceanProject instance
            grid: GridSpec instance
            labels: LabelsStore instance
        """
        self.project = project
        self.grid = grid
        self.labels = labels

        # Initialize preset manager
        if self.project:
            self.preset_manager = PresetManager(self.project)
            self.refresh_preset_list()

        # Update dataset information display
        self.update_dataset_info()
        
        # Update augmentation metrics
        if hasattr(self, 'aug_metrics_label'):
            self.update_augmentation_metrics()
        
        # Refresh GPU devices list (in case project was opened on different machine)
        if hasattr(self, 'gpu_device_combo'):
            self.populate_gpu_devices()

        # Enable start button if project has labeled data
        self.update_ui_state()
        
        # For testing: Auto-apply suggested parameters and data augmentation settings
        self.auto_apply_test_settings()

    def refresh_preset_list(self):
        """Refresh the preset dropdown list."""
        if not self.preset_manager:
            return

        self.preset_combo.clear()
        presets = self.preset_manager.list_presets()
        self.preset_combo.addItems(presets)

    def on_preset_selected(self, preset_name: str):
        """Handle preset selection (just updates UI, doesn't load)."""
        # This is just for display - actual loading happens on "Load" button
        pass

    def on_load_preset(self):
        """Load the selected preset and populate parameter fields."""
        if not self.preset_manager:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        preset_name = self.preset_combo.currentText()
        if not preset_name:
            QMessageBox.warning(self, "Error", "Please select a preset.")
            return

        try:
            config = self.preset_manager.load_preset(preset_name)
            self.current_config = config
            # Populate architecture fields
            self.set_architecture_config(config.architecture)
            QMessageBox.information(
                self,
                "Preset Loaded",
                f"Preset '{preset_name}' loaded successfully.",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load preset:\n{str(e)}"
            )

    def on_export_config(self):
        """Export current configuration to JSON file."""
        if not self.current_config:
            QMessageBox.warning(
                self, "Error", "No configuration to export. Please create or load a configuration first."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Training Configuration",
            "",
            "JSON Files (*.json);;All Files (*)",
        )

        if file_path:
            try:
                self.current_config.save(file_path)
                QMessageBox.information(
                    self, "Success", f"Configuration exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to export configuration:\n{str(e)}"
                )

    def on_import_config(self):
        """Import configuration from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Training Configuration",
            "",
            "JSON Files (*.json);;All Files (*)",
        )

        if file_path:
            try:
                config = TrainingConfig.load(file_path)
                config.validate()
                self.current_config = config
                # Populate architecture fields
                self.set_architecture_config(config.architecture)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Configuration imported successfully.",
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to import configuration:\n{str(e)}"
                )

    def on_auto_suggest(self):
        """Auto-suggest parameters based on dataset analysis."""
        if not self.project:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        try:
            suggester = ParameterSuggester(self.project)
            config = suggester.suggest_full_config()
            self.current_config = config

            # Get dataset stats for info message
            analysis = suggester.analyze_dataset()

            # Populate architecture fields
            self.set_architecture_config(config.architecture)

            QMessageBox.information(
                self,
                "Parameters Suggested",
                f"Parameters suggested based on:\n"
                f"- {analysis.total_labeled_tiles} labeled tiles\n"
                f"- {analysis.num_classes} classes",
            )
        except ValueError as e:
            QMessageBox.warning(self, "Cannot Suggest", str(e))
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to suggest parameters:\n{str(e)}"
            )

    def update_ui_state(self):
        """Update UI state based on project status."""
        if not self.project:
            self.start_button.setEnabled(False)
            return

        # Check if project has labeled data
        # This will be implemented when we add dataset stats
        # For now, enable if project exists
        self.start_button.setEnabled(True)
    
    def auto_apply_test_settings(self):
        """Auto-apply suggested parameters and data augmentation settings for testing."""
        if not self.project:
            return
        
        try:
            # Auto-apply suggested parameters
            suggester = ParameterSuggester(self.project)
            config = suggester.suggest_full_config()
            self.current_config = config
            
            # Populate architecture fields
            self.set_architecture_config(config.architecture)
            
            # Populate hyperparameters from suggested config
            if hasattr(self, 'batch_size_spin'):
                self.batch_size_spin.setValue(config.batch_size)
            if hasattr(self, 'num_epochs_spin'):
                self.num_epochs_spin.setValue(config.num_epochs)
            if hasattr(self, 'learning_rate_spin'):
                self.learning_rate_spin.setValue(config.learning_rate)
            if hasattr(self, 'seed_spin') and config.seed:
                self.seed_spin.setValue(config.seed)
            
            # Set optimizer
            if hasattr(self, 'optimizer_combo'):
                self.optimizer_combo.setCurrentText(config.optimizer)
                self.on_optimizer_changed()  # Update optimizer params widgets
                # Set optimizer parameters
                if "weight_decay" in config.optimizer_params and "weight_decay" in self.optimizer_param_widgets:
                    self.optimizer_param_widgets["weight_decay"].setValue(config.optimizer_params["weight_decay"])
                if "momentum" in config.optimizer_params and "momentum" in self.optimizer_param_widgets:
                    self.optimizer_param_widgets["momentum"].setValue(config.optimizer_params["momentum"])
            
            # Set loss function
            if hasattr(self, 'loss_function_combo'):
                self.loss_function_combo.setCurrentText(config.loss_function)
            
            # Set validation split and class weights
            if hasattr(self, 'validation_split_spin'):
                self.validation_split_spin.setValue(config.validation_split)
            if hasattr(self, 'class_weights_checkbox'):
                self.class_weights_checkbox.setChecked(config.class_weights)
            
            # Set GPU settings
            if hasattr(self, 'use_gpu_checkbox'):
                self.use_gpu_checkbox.setChecked(config.use_gpu)
                self.on_gpu_checkbox_toggled(config.use_gpu)
            
            # Set early stopping
            if hasattr(self, 'early_stopping_checkbox'):
                if config.early_stopping:
                    self.early_stopping_checkbox.setChecked(True)
                    if hasattr(self, 'early_stopping_patience_spin'):
                        self.early_stopping_patience_spin.setValue(config.early_stopping.get("patience", 10))
                    if hasattr(self, 'early_stopping_min_delta_spin'):
                        self.early_stopping_min_delta_spin.setValue(config.early_stopping.get("min_delta", 0.0))
                else:
                    self.early_stopping_checkbox.setChecked(False)
            
            # Set checkpoint frequency
            if hasattr(self, 'save_checkpoint_every_spin'):
                self.save_checkpoint_every_spin.setValue(config.save_checkpoint_every)
            
            # Auto-apply data augmentation settings from image:
            # - Random Horizontal Flip: enabled
            # Data augmentation defaults removed - all checkboxes remain unchecked
            
            print("Auto-applied suggested parameters for testing")
        except Exception as e:
            # Silently fail if auto-apply doesn't work (e.g., no labels yet)
            import traceback
            print(f"Could not auto-apply test settings: {e}")
            traceback.print_exc()

    def highlight_invalid_field(self, widget, is_invalid: bool):
        """Highlight a widget with red border if invalid."""
        if is_invalid:
            widget.setStyleSheet("border: 2px solid red;")
        else:
            widget.setStyleSheet("")  # Reset to default
    
    def validate_training_parameters(self) -> tuple[bool, str, Optional[str]]:
        """
        Validate all training parameters before starting training.
        
        Returns:
            Tuple of (is_valid, error_message, invalid_field_name)
            invalid_field_name can be used to highlight the problematic field
        """
        # Check project is loaded
        if not self.project:
            return False, "No project loaded. Please open or create a project first.", None
        
        # Check project has labeled data
        if not self.labels:
            return False, "No labels found. Please ensure the project has a labels.csv file.", None
        
        # Count labeled tiles (non-empty labels)
        labeled_tiles = [
            (img, i, j, label)
            for img, i, j, label in self.labels.iter_rows()
            if label and label.strip()  # Only count non-empty labels
        ]
        if len(labeled_tiles) == 0:
            return False, "No labeled data available. Please label some tiles in the Label tab first.", None
        
        # Check classes are defined
        # Get classes from labels (actual classes used) rather than classes.json
        # This ensures we validate against what's actually in the data
        try:
            # Get unique classes from labels
            classes_in_labels = set()
            for _, _, _, label in self.labels.iter_rows():
                if label and label.strip():
                    classes_in_labels.add(label.strip())
            
            if len(classes_in_labels) == 0:
                return False, "No labeled data found. Please label some tiles first.", None
            
            classes = sorted(list(classes_in_labels))
            
            # Update classes.json to match actual classes in labels (if mismatch)
            try:
                import json
                classes_path = self.project.paths.configs_dir / "classes.json"
                if classes_path.exists():
                    try:
                        classes_from_json = json.loads(classes_path.read_text())
                        if not isinstance(classes_from_json, list):
                            classes_from_json = []
                    except Exception:
                        classes_from_json = []
                    
                    # Check if classes.json needs updating
                    if set(classes_from_json) != set(classes):
                        # Update classes.json to match actual classes
                        classes_path.write_text(json.dumps(classes, indent=2) + "\n")
                        print(f"Updated classes.json to match labels: {classes}")
                else:
                    # Create classes.json if it doesn't exist
                    classes_path.write_text(json.dumps(classes, indent=2) + "\n")
                    print(f"Created classes.json with classes from labels: {classes}")
            except Exception as e:
                print(f"Warning: Could not update classes.json: {e}")
        except Exception as e:
            return False, f"Failed to determine classes from labels: {str(e)}", None
        
        # Validate architecture
        try:
            arch_config = self.get_architecture_config()
            if arch_config is None:
                return False, "Invalid model architecture. Please check your layer configuration.", "layers_table"
            
            # Validate architecture config
            arch_config.validate()
            
            # Check number of classes matches - auto-fix if needed
            if arch_config.num_classes != len(classes):
                # Auto-fix: Update num_classes to match actual classes
                self.num_classes_spin.setValue(len(classes))
                print(f"Auto-updated number of classes to {len(classes)} to match labeled data")
                # Re-get config with updated num_classes
                arch_config = self.get_architecture_config()
            
            self.highlight_invalid_field(self.num_classes_spin, False)
            
            # Check input size matches grid
            if self.grid:
                if arch_config.input_size != (self.grid.tile_w, self.grid.tile_h):
                    return False, (
                        f"Input size mismatch: Architecture expects {arch_config.input_size}, "
                        f"but grid tile size is ({self.grid.tile_w}, {self.grid.tile_h}). "
                        f"Input size is automatically set from grid and cannot be changed."
                    ), None
        except ValueError as e:
            return False, f"Architecture validation failed: {str(e)}", "layers_table"
        except Exception as e:
            return False, f"Failed to validate architecture: {str(e)}", None
        
        # Validate hyperparameters
        try:
            batch_size = self.batch_size_spin.value()
            if batch_size <= 0:
                self.highlight_invalid_field(self.batch_size_spin, True)
                return False, f"Batch size must be greater than 0, got {batch_size}.", "batch_size_spin"
            self.highlight_invalid_field(self.batch_size_spin, False)
            
            num_epochs = self.num_epochs_spin.value()
            if num_epochs <= 0:
                self.highlight_invalid_field(self.num_epochs_spin, True)
                return False, f"Number of epochs must be greater than 0, got {num_epochs}.", "num_epochs_spin"
            self.highlight_invalid_field(self.num_epochs_spin, False)
            
            learning_rate = self.learning_rate_spin.value()
            if learning_rate <= 0:
                self.highlight_invalid_field(self.learning_rate_spin, True)
                return False, f"Learning rate must be greater than 0, got {learning_rate}.", "learning_rate_spin"
            if learning_rate >= 1.0:
                self.highlight_invalid_field(self.learning_rate_spin, True)
                return False, f"Learning rate is unusually high ({learning_rate}). Typical values are between 1e-6 and 1e-1.", "learning_rate_spin"
            self.highlight_invalid_field(self.learning_rate_spin, False)
            
            validation_split = self.validation_split_spin.value()
            if validation_split < 0.0 or validation_split > 0.5:
                self.highlight_invalid_field(self.validation_split_spin, True)
                return False, f"Validation split must be between 0.0 and 0.5, got {validation_split}.", "validation_split_spin"
            self.highlight_invalid_field(self.validation_split_spin, False)
            
            # Check if validation split would leave enough training data
            if validation_split > 0 and len(labeled_tiles) * (1 - validation_split) < batch_size:
                self.highlight_invalid_field(self.validation_split_spin, True)
                self.highlight_invalid_field(self.batch_size_spin, True)
                return False, (
                    f"Validation split ({validation_split:.1%}) would leave fewer training samples "
                    f"({int(len(labeled_tiles) * (1 - validation_split))}) than batch size ({batch_size}). "
                    f"Please reduce validation split or batch size."
                ), "validation_split_spin"
            
            # Check batch size doesn't exceed dataset size
            if batch_size > len(labeled_tiles):
                self.highlight_invalid_field(self.batch_size_spin, True)
                return False, (
                    f"Batch size ({batch_size}) exceeds total labeled tiles ({len(labeled_tiles)}). "
                    f"Please reduce batch size."
                ), "batch_size_spin"
            
        except Exception as e:
            return False, f"Failed to validate hyperparameters: {str(e)}", None
        
        # Validate GPU availability if GPU is requested
        if self.use_gpu_checkbox.isChecked():
            try:
                import torch
                if not torch.cuda.is_available():
                    device_name = self.get_selected_device()
                    if device_name.startswith("cuda"):
                        self.highlight_invalid_field(self.use_gpu_checkbox, True)
                        return False, (
                            "GPU training requested but CUDA is not available. "
                            "Please install CUDA-enabled PyTorch or uncheck 'Use GPU'."
                        ), "use_gpu_checkbox"
            except ImportError:
                return False, "PyTorch is not installed. Please install PyTorch to use GPU training.", None
            except Exception as e:
                return False, f"Failed to check GPU availability: {str(e)}", None
        
        # Validate early stopping parameters if enabled
        if self.early_stopping_checkbox.isChecked():
            patience = self.early_stopping_patience_spin.value()
            if patience <= 0:
                self.highlight_invalid_field(self.early_stopping_patience_spin, True)
                return False, f"Early stopping patience must be greater than 0, got {patience}.", "early_stopping_patience_spin"
            if patience >= num_epochs:
                self.highlight_invalid_field(self.early_stopping_patience_spin, True)
                return False, (
                    f"Early stopping patience ({patience}) must be less than number of epochs ({num_epochs})."
                ), "early_stopping_patience_spin"
            self.highlight_invalid_field(self.early_stopping_patience_spin, False)
        
        # Validate checkpoint saving frequency
        save_checkpoint_every = self.save_checkpoint_every_spin.value()
        if save_checkpoint_every <= 0:
            self.highlight_invalid_field(self.save_checkpoint_every_spin, True)
            return False, f"Checkpoint save frequency must be greater than 0, got {save_checkpoint_every}.", "save_checkpoint_every_spin"
        self.highlight_invalid_field(self.save_checkpoint_every_spin, False)
        
        return True, "", None

    def on_start_training(self):
        """Handle start training button click."""
        # Comprehensive parameter validation
        is_valid, error_msg, invalid_field = self.validate_training_parameters()
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_msg)
            # Scroll to invalid field if possible
            if invalid_field and hasattr(self, invalid_field):
                widget = getattr(self, invalid_field)
                if widget:
                    widget.setFocus()
            return
        
        # Validate parameters and create config
        try:
            config = self.get_training_config()
            if config is None:
                QMessageBox.warning(self, "Error", "Invalid training configuration. Please check your parameters.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create training config:\n{str(e)}")
            return
        
        # Create training run
        try:
            self.current_run = create_run(self.project)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create training run:\n{str(e)}")
            return
        
        # Clear previous plots
        if hasattr(self, 'training_plot'):
            self.training_plot.clear()
            self.training_plot.set_max_epochs(config.num_epochs)
        
        # Clean up any existing training thread before creating a new one
        if hasattr(self, 'training_thread') and self.training_thread is not None:
            if self.training_thread.isRunning():
                self.training_thread.stop()
                self.training_thread.wait(3000)  # Wait up to 3 seconds for thread to finish
                if self.training_thread.isRunning():
                    print("Warning: Previous training thread did not stop gracefully")
            self.training_thread.deleteLater()
        
        # Create and start training thread
        self.training_thread = TrainingThread(config, self.project, self.current_run)
        # Set parent to ensure thread is kept alive
        self.training_thread.setParent(self)
        
        # Connect signals
        self.training_thread.epoch_completed.connect(self.on_epoch_completed)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.training_error.connect(self.on_training_error)
        self.training_thread.status_update.connect(self.on_status_update)
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_model_button.setEnabled(False)
        
        # Disable parameter inputs during training
        self.set_parameters_enabled(False)
        
        # Start training
        self.training_thread.start()
        self.status_label.setText("Status: Starting training...")

    def on_stop_training(self):
        """Handle stop training button click."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.status_label.setText("Status: Stopping training...")
            self.stop_button.setEnabled(False)  # Disable while stopping

    def on_save_model(self):
        """Handle save model button click."""
        if not self.current_run:
            QMessageBox.warning(self, "Error", "No training run available to save.")
            return
        
        # Open file dialog to save model
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            str(self.current_run.run_dir / "model.pth"),
            "PyTorch Model (*.pth);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load best checkpoint and save to user location
                checkpoint, epoch, metrics = self.current_run.load_checkpoint(best=True)
                import torch
                torch.save(checkpoint, file_path)
                QMessageBox.information(self, "Success", f"Model saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model:\n{str(e)}")
    
    def on_epoch_completed(self, epoch: int, metrics: dict):
        """Handle epoch completion signal - update graphs."""
        train_loss = metrics.get("loss", 0.0)
        val_loss = metrics.get("val_loss")
        train_acc = metrics.get("accuracy")
        val_acc = metrics.get("val_accuracy")
        
        if hasattr(self, 'training_plot') and self.training_plot is not None:
            print(f"Updating training plot for epoch {epoch}")
            self.training_plot.update_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc
            )
        else:
            print(f"WARNING: training_plot not found or None")
        
        # Update status
        status_text = f"Epoch {epoch}/{self.current_config.num_epochs if self.current_config else '?'}"
        if train_loss is not None:
            status_text += f" | Loss: {train_loss:.4f}"
        if val_loss is not None:
            status_text += f" | Val Loss: {val_loss:.4f}"
        if train_acc is not None:
            status_text += f" | Acc: {train_acc:.2%}"
        self.status_label.setText(f"Status: {status_text}")
    
    def on_training_finished(self, history: TrainingHistory):
        """Handle training finished signal."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_model_button.setEnabled(True)
        
        # Re-enable parameter inputs
        self.set_parameters_enabled(True)
        
        # Update status
        final_loss = history.train_loss[-1] if history.train_loss else 0.0
        final_acc = history.train_accuracy[-1] if history.train_accuracy else 0.0
        self.status_label.setText(
            f"Status: Training completed! Final Loss: {final_loss:.4f}, Accuracy: {final_acc:.2%}"
        )
        
        QMessageBox.information(
            self,
            "Training Complete",
            f"Training finished successfully!\n\n"
            f"Epochs: {len(history.epochs)}\n"
            f"Final Loss: {final_loss:.4f}\n"
            f"Final Accuracy: {final_acc:.2%}\n\n"
            f"Run ID: {self.current_run.run_id}"
        )
    
    def on_training_error(self, error_msg: str):
        """Handle training error signal."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_model_button.setEnabled(False)
        
        # Re-enable parameter inputs
        self.set_parameters_enabled(True)
        
        self.status_label.setText("Status: Training failed")
        QMessageBox.critical(self, "Training Error", error_msg)
    
    def on_status_update(self, status: str):
        """Handle status update signal."""
        self.status_label.setText(f"Status: {status}")
    
    def set_parameters_enabled(self, enabled: bool):
        """Enable or disable all parameter input widgets."""
        # This will be implemented to disable UI during training
        # For now, just a placeholder
        pass
    
    def get_training_config(self) -> Optional[TrainingConfig]:
        """Get TrainingConfig from current UI state."""
        try:
            arch_config = self.get_architecture_config()
            if arch_config is None:
                return None
            
            # Get hyperparameters
            batch_size = self.batch_size_spin.value()
            num_epochs = self.num_epochs_spin.value()
            learning_rate = self.learning_rate_spin.value()
            seed = self.seed_spin.value() if self.seed_spin.value() > 0 else None
            
            # Optimizer
            optimizer = self.optimizer_combo.currentText()
            optimizer_params = {}
            if optimizer in ["adam", "adamw"]:
                if "weight_decay" in self.optimizer_param_widgets:
                    optimizer_params["weight_decay"] = self.optimizer_param_widgets["weight_decay"].value()
                else:
                    optimizer_params["weight_decay"] = 0.0
            elif optimizer == "sgd":
                if "momentum" in self.optimizer_param_widgets:
                    optimizer_params["momentum"] = self.optimizer_param_widgets["momentum"].value()
                else:
                    optimizer_params["momentum"] = 0.9
                if "weight_decay" in self.optimizer_param_widgets:
                    optimizer_params["weight_decay"] = self.optimizer_param_widgets["weight_decay"].value()
                else:
                    optimizer_params["weight_decay"] = 0.0
            
            # Loss function
            loss_function = self.loss_function_combo.currentText()
            
            # Data settings
            validation_split = self.validation_split_spin.value()
            class_weights = self.class_weights_checkbox.isChecked()
            
            # Data augmentation
            data_augmentation = {
                "horizontal_flip": self.aug_flip_checkbox.isChecked(),
                "rotation": self.aug_rotate_checkbox.isChecked(),
                "rotation_angle": self.aug_rotate_angle_spin.value() if self.aug_rotate_checkbox.isChecked() else 0,
                "brightness": self.aug_brightness_checkbox.isChecked(),
                "brightness_min": self.aug_brightness_min_spin.value() if self.aug_brightness_checkbox.isChecked() else 0.8,
                "brightness_max": self.aug_brightness_max_spin.value() if self.aug_brightness_checkbox.isChecked() else 1.2,
            }
            
            # Training options
            use_gpu = self.use_gpu_checkbox.isChecked()
            device = self.get_selected_device() if use_gpu else "cpu"
            
            early_stopping = None
            if self.early_stopping_checkbox.isChecked():
                early_stopping = {
                    "patience": self.early_stopping_patience_spin.value(),
                    "min_delta": self.early_stopping_min_delta_spin.value()
                }
            
            save_checkpoint_every = self.save_checkpoint_every_spin.value()
            
            # Create config
            config = TrainingConfig(
                architecture=arch_config,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                loss_function=loss_function,
                use_gpu=use_gpu,
                device=device,
                validation_split=validation_split,
                data_augmentation=data_augmentation,
                early_stopping=early_stopping,
                class_weights=class_weights,
                seed=seed,
                save_checkpoint_every=save_checkpoint_every
            )
            
            config.validate()
            self.current_config = config
            return config
            
        except Exception as e:
            print(f"Error creating training config: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_hyperparameters_group(self) -> QGroupBox:
        """Create the training hyperparameters group box."""
        group = QGroupBox("Training Hyperparameters")
        layout = QVBoxLayout()
        group.setLayout(layout)

        # ===== Hyperparameters on Different Rows =====
        
        # Basic Parameters
        basic_params_widget = QWidget()
        basic_layout = QVBoxLayout()
        basic_params_widget.setLayout(basic_layout)
        basic_layout.addWidget(QLabel("Basic Parameters:"))
        
        basic_inner = QHBoxLayout()
        basic_inner.addWidget(QLabel("Batch:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(1024)
        self.batch_size_spin.setValue(32)
        self.batch_size_spin.setSingleStep(8)
        basic_inner.addWidget(self.batch_size_spin)
        
        basic_inner.addWidget(QLabel("Epochs:"))
        self.num_epochs_spin = QSpinBox()
        self.num_epochs_spin.setMinimum(1)
        self.num_epochs_spin.setMaximum(10000)
        self.num_epochs_spin.setValue(50)
        basic_inner.addWidget(self.num_epochs_spin)
        
        basic_inner.addWidget(QLabel("LR:"))
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setMinimum(1e-6)
        self.learning_rate_spin.setMaximum(1.0)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setDecimals(6)
        basic_inner.addWidget(self.learning_rate_spin)
        
        basic_inner.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setMinimum(0)
        self.seed_spin.setMaximum(2147483647)
        self.seed_spin.setValue(0)
        self.seed_spin.setSpecialValueText("None")
        basic_inner.addWidget(self.seed_spin)
        basic_inner.addStretch()
        basic_layout.addLayout(basic_inner)
        layout.addWidget(basic_params_widget)
        
        # Optimizer
        optimizer_widget = QWidget()
        optimizer_layout = QVBoxLayout()
        optimizer_widget.setLayout(optimizer_layout)
        optimizer_layout.addWidget(QLabel("Optimizer:"))
        
        opt_inner = QHBoxLayout()
        opt_inner.addWidget(QLabel("Type:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd", "rmsprop", "adamw"])
        self.optimizer_combo.setCurrentText("adam")
        self.optimizer_combo.currentTextChanged.connect(self.on_optimizer_changed)
        opt_inner.addWidget(self.optimizer_combo)
        
        # Optimizer parameters container (will be populated dynamically)
        self.optimizer_params_widget = QWidget()
        self.optimizer_params_layout = QHBoxLayout()
        self.optimizer_params_widget.setLayout(self.optimizer_params_layout)
        opt_inner.addWidget(self.optimizer_params_widget)
        opt_inner.addStretch()
        
        # Initialize optimizer parameter widgets
        self.optimizer_param_widgets = {}
        self.create_optimizer_params_widgets()
        
        optimizer_layout.addLayout(opt_inner)
        
        # Optimizer and Loss on same row
        opt_loss_row = QHBoxLayout()
        opt_loss_row.addWidget(optimizer_widget)
        
        # Loss Function
        loss_widget = QWidget()
        loss_layout = QVBoxLayout()
        loss_widget.setLayout(loss_layout)
        loss_layout.addWidget(QLabel("Loss:"))
        
        loss_inner = QHBoxLayout()
        self.loss_function_combo = QComboBox()
        self.loss_function_combo.addItems(["cross_entropy", "focal_loss", "nll_loss"])
        self.loss_function_combo.setCurrentText("cross_entropy")
        loss_inner.addWidget(self.loss_function_combo)
        loss_inner.addStretch()
        loss_layout.addLayout(loss_inner)
        opt_loss_row.addWidget(loss_widget)
        
        opt_loss_row.addStretch()
        layout.addLayout(opt_loss_row)
        
        # Training Options
        options_widget = QWidget()
        options_layout = QVBoxLayout()
        options_widget.setLayout(options_layout)
        options_layout.addWidget(QLabel("Options:"))
        
        options_inner = QHBoxLayout()
        self.use_gpu_checkbox = QCheckBox("GPU")
        self.use_gpu_checkbox.setChecked(True)
        options_inner.addWidget(self.use_gpu_checkbox)
        
        options_inner.addWidget(QLabel("Device:"))
        self.gpu_device_combo = QComboBox()
        self.gpu_device_combo.setEnabled(True)
        options_inner.addWidget(self.gpu_device_combo)
        
        self.early_stopping_checkbox = QCheckBox("Early Stop")
        self.early_stopping_checkbox.setChecked(False)
        self.early_stopping_checkbox.toggled.connect(self.on_early_stopping_toggled)
        options_inner.addWidget(self.early_stopping_checkbox)
        
        options_inner.addWidget(QLabel("Checkpoint:"))
        self.save_checkpoint_every_spin = QSpinBox()
        self.save_checkpoint_every_spin.setMinimum(1)
        self.save_checkpoint_every_spin.setMaximum(1000)
        self.save_checkpoint_every_spin.setValue(5)
        options_inner.addWidget(self.save_checkpoint_every_spin)
        options_inner.addStretch()
        
        # Populate GPU devices after checkbox is created
        self.populate_gpu_devices()
        # Connect checkbox toggle after population
        self.use_gpu_checkbox.toggled.connect(self.on_gpu_checkbox_toggled)
        
        options_layout.addLayout(options_inner)
        layout.addWidget(options_widget)

        # ===== Data Settings =====
        data_group = QGroupBox("Data Settings")
        data_layout = QVBoxLayout()
        data_group.setLayout(data_layout)

        # Validation split and Class weights on same row
        val_class_row = QHBoxLayout()
        val_class_row.addWidget(QLabel("Validation Split:"))
        self.validation_split_spin = QDoubleSpinBox()
        self.validation_split_spin.setMinimum(0.0)
        self.validation_split_spin.setMaximum(0.5)
        self.validation_split_spin.setValue(0.2)
        self.validation_split_spin.setSingleStep(0.05)
        self.validation_split_spin.setDecimals(2)
        val_class_row.addWidget(self.validation_split_spin)
        val_class_row.addWidget(QLabel("(0.0 - 0.5)"))
        
        self.class_weights_checkbox = QCheckBox("Use Class Weights")
        self.class_weights_checkbox.setChecked(False)
        val_class_row.addWidget(self.class_weights_checkbox)
        val_class_row.addStretch()
        data_layout.addLayout(val_class_row)

        layout.addWidget(data_group)

        # Early stopping parameters (shown when enabled, added to hyperparams row if needed)
        early_stop_params_widget = QWidget()
        early_stop_layout = QHBoxLayout()
        early_stop_params_widget.setLayout(early_stop_layout)
        
        early_stop_layout.addWidget(QLabel("Patience:"))
        self.early_stopping_patience_spin = QSpinBox()
        self.early_stopping_patience_spin.setMinimum(1)
        self.early_stopping_patience_spin.setMaximum(1000)
        self.early_stopping_patience_spin.setValue(10)
        self.early_stopping_patience_spin.setEnabled(False)
        early_stop_layout.addWidget(self.early_stopping_patience_spin)
        
        early_stop_layout.addWidget(QLabel("Min Delta:"))
        self.early_stopping_min_delta_spin = QDoubleSpinBox()
        self.early_stopping_min_delta_spin.setMinimum(0.0)
        self.early_stopping_min_delta_spin.setMaximum(1.0)
        self.early_stopping_min_delta_spin.setValue(0.0)
        self.early_stopping_min_delta_spin.setDecimals(6)
        self.early_stopping_min_delta_spin.setEnabled(False)
        early_stop_layout.addWidget(self.early_stopping_min_delta_spin)
        early_stop_layout.addStretch()

        self.early_stopping_params_widget = early_stop_params_widget
        self.early_stopping_params_widget.setEnabled(False)
        # Add early stopping params below the hyperparams row when enabled
        layout.addWidget(self.early_stopping_params_widget)

        return group

    def create_optimizer_params_widgets(self):
        """Create optimizer parameter widgets based on selected optimizer."""
        # Clear existing widgets
        while self.optimizer_params_layout.count():
            item = self.optimizer_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.optimizer_param_widgets.clear()

        optimizer = self.optimizer_combo.currentText()

        if optimizer == "adam" or optimizer == "adamw":
            # Weight decay (horizontal layout)
            self.optimizer_params_layout.addWidget(QLabel("Weight Decay:"))
            weight_decay_spin = QDoubleSpinBox()
            weight_decay_spin.setMinimum(0.0)
            weight_decay_spin.setMaximum(1.0)
            weight_decay_spin.setValue(0.0)
            weight_decay_spin.setSingleStep(0.0001)
            weight_decay_spin.setDecimals(6)
            weight_decay_spin.setMaximumWidth(100)
            self.optimizer_params_layout.addWidget(weight_decay_spin)
            self.optimizer_param_widgets["weight_decay"] = weight_decay_spin

        elif optimizer == "sgd":
            # Momentum (horizontal layout)
            self.optimizer_params_layout.addWidget(QLabel("Momentum:"))
            momentum_spin = QDoubleSpinBox()
            momentum_spin.setMinimum(0.0)
            momentum_spin.setMaximum(1.0)
            momentum_spin.setValue(0.9)
            momentum_spin.setSingleStep(0.01)
            momentum_spin.setDecimals(2)
            momentum_spin.setMaximumWidth(100)
            self.optimizer_params_layout.addWidget(momentum_spin)
            self.optimizer_param_widgets["momentum"] = momentum_spin

            # Weight decay (horizontal layout)
            self.optimizer_params_layout.addWidget(QLabel("Weight Decay:"))
            weight_decay_spin = QDoubleSpinBox()
            weight_decay_spin.setMinimum(0.0)
            weight_decay_spin.setMaximum(1.0)
            weight_decay_spin.setValue(0.0)
            weight_decay_spin.setSingleStep(0.0001)
            weight_decay_spin.setDecimals(6)
            weight_decay_spin.setMaximumWidth(100)
            self.optimizer_params_layout.addWidget(weight_decay_spin)
            self.optimizer_param_widgets["weight_decay"] = weight_decay_spin

        # rmsprop has no additional parameters typically

    def create_data_augmentation_group(self) -> QGroupBox:
        """Create the data augmentation group box with before/after metrics."""
        aug_group = QGroupBox("Data Augmentation")
        aug_layout = QVBoxLayout()
        aug_group.setLayout(aug_layout)

        # Augmentation checkboxes
        self.aug_flip_checkbox = QCheckBox("Random Horizontal Flip")
        self.aug_flip_checkbox.setChecked(False)
        self.aug_flip_checkbox.toggled.connect(self.update_augmentation_metrics)
        aug_layout.addWidget(self.aug_flip_checkbox)

        self.aug_rotate_checkbox = QCheckBox("Random Rotation")
        self.aug_rotate_checkbox.setChecked(False)
        self.aug_rotate_checkbox.toggled.connect(self.update_augmentation_metrics)
        aug_layout.addWidget(self.aug_rotate_checkbox)

        # Rotation angle
        rotate_angle_row = QHBoxLayout()
        rotate_angle_row.addWidget(QLabel("Max Angle (deg):"))
        self.aug_rotate_angle_spin = QSpinBox()
        self.aug_rotate_angle_spin.setMinimum(0)
        self.aug_rotate_angle_spin.setMaximum(180)
        self.aug_rotate_angle_spin.setValue(15)
        self.aug_rotate_angle_spin.setEnabled(False)
        self.aug_rotate_checkbox.toggled.connect(self.aug_rotate_angle_spin.setEnabled)
        self.aug_rotate_angle_spin.valueChanged.connect(self.update_augmentation_metrics)
        rotate_angle_row.addWidget(self.aug_rotate_angle_spin)
        rotate_angle_row.addStretch()
        aug_layout.addLayout(rotate_angle_row)

        self.aug_brightness_checkbox = QCheckBox("Random Brightness")
        self.aug_brightness_checkbox.setChecked(False)
        self.aug_brightness_checkbox.toggled.connect(self.update_augmentation_metrics)
        aug_layout.addWidget(self.aug_brightness_checkbox)

        # Brightness factor
        brightness_row = QHBoxLayout()
        brightness_row.addWidget(QLabel("Range:"))
        self.aug_brightness_min_spin = QDoubleSpinBox()
        self.aug_brightness_min_spin.setMinimum(0.1)
        self.aug_brightness_min_spin.setMaximum(1.0)
        self.aug_brightness_min_spin.setValue(0.8)
        self.aug_brightness_min_spin.setDecimals(2)
        self.aug_brightness_min_spin.setEnabled(False)
        self.aug_brightness_checkbox.toggled.connect(self.aug_brightness_min_spin.setEnabled)
        self.aug_brightness_min_spin.valueChanged.connect(self.update_augmentation_metrics)
        brightness_row.addWidget(QLabel("Min:"))
        brightness_row.addWidget(self.aug_brightness_min_spin)
        
        self.aug_brightness_max_spin = QDoubleSpinBox()
        self.aug_brightness_max_spin.setMinimum(1.0)
        self.aug_brightness_max_spin.setMaximum(2.0)
        self.aug_brightness_max_spin.setValue(1.2)
        self.aug_brightness_max_spin.setDecimals(2)
        self.aug_brightness_max_spin.setEnabled(False)
        self.aug_brightness_checkbox.toggled.connect(self.aug_brightness_max_spin.setEnabled)
        self.aug_brightness_max_spin.valueChanged.connect(self.update_augmentation_metrics)
        brightness_row.addWidget(QLabel("Max:"))
        brightness_row.addWidget(self.aug_brightness_max_spin)
        brightness_row.addStretch()
        aug_layout.addLayout(brightness_row)

        # Before/After metrics
        metrics_label = QLabel("Augmentation Impact:")
        metrics_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        aug_layout.addWidget(metrics_label)
        
        self.aug_metrics_label = QLabel("No augmentation enabled")
        self.aug_metrics_label.setWordWrap(True)
        self.aug_metrics_label.setStyleSheet("color: #666; font-size: 10px;")
        aug_layout.addWidget(self.aug_metrics_label)

        return aug_group

    def update_augmentation_metrics(self):
        """Update the before/after metrics display for data augmentation."""
        if not self.project or not self.grid or not self.labels:
            if hasattr(self, 'aug_metrics_label'):
                self.aug_metrics_label.setText("No project loaded")
            return

        try:
            from ofc.core import get_dataset_stats
            stats: DatasetStats = get_dataset_stats(self.project, self.grid, self.labels)
            
            # Calculate base metrics
            base_tiles = stats.total_labeled_tiles
            base_classes = stats.num_classes
            
            # Calculate augmentation multiplier
            aug_multiplier = 1.0
            aug_details = []
            
            if self.aug_flip_checkbox.isChecked():
                aug_multiplier *= 2.0  # Horizontal flip doubles the data
                aug_details.append("Flip: 2x")
            
            if self.aug_rotate_checkbox.isChecked():
                # Rotation creates multiple variations
                angle = self.aug_rotate_angle_spin.value()
                if angle > 0:
                    # Approximate: rotation creates ~(360/angle) variations
                    rotations = max(2, int(360 / max(angle, 1)))
                    aug_multiplier *= rotations
                    aug_details.append(f"Rotation: ~{rotations}x")
            
            if self.aug_brightness_checkbox.isChecked():
                # Brightness creates continuous variations, estimate 3x
                aug_multiplier *= 3.0
                aug_details.append("Brightness: ~3x")
            
            if aug_multiplier > 1.0:
                estimated_tiles = int(base_tiles * aug_multiplier)
                metrics_text = f"Before: {base_tiles} tiles, {base_classes} classes\n"
                metrics_text += f"After: ~{estimated_tiles} tiles (×{aug_multiplier:.1f})\n"
                metrics_text += f"Details: {', '.join(aug_details)}"
                self.aug_metrics_label.setText(metrics_text)
            else:
                self.aug_metrics_label.setText("No augmentation enabled")
                
        except Exception as e:
            if hasattr(self, 'aug_metrics_label'):
                self.aug_metrics_label.setText(f"Error calculating metrics: {str(e)}")

    def on_optimizer_changed(self):
        """Handle optimizer type change - update parameter widgets."""
        self.create_optimizer_params_widgets()

    def on_early_stopping_toggled(self, checked: bool):
        """Handle early stopping checkbox toggle."""
        self.early_stopping_params_widget.setEnabled(checked)
        self.early_stopping_patience_spin.setEnabled(checked)
        self.early_stopping_min_delta_spin.setEnabled(checked)

    def populate_gpu_devices(self):
        """Populate GPU device dropdown with available devices using device utility."""
        if not hasattr(self, 'gpu_device_combo'):
            return  # Widget not created yet
        
        self.gpu_device_combo.clear()
        
        try:
            # Use device utility to list available devices
            devices = list_available_devices()
            
            for device_info in devices:
                name = device_info["name"]
                device_str = device_info["device"]
                memory_gb = device_info.get("memory_gb")
                
                # Format display name with memory if available
                if memory_gb is not None:
                    display_name = f"{name} ({memory_gb:.1f} GB)"
                else:
                    display_name = name
                
                self.gpu_device_combo.addItem(display_name, device_str)
            
            # Set default selection based on checkbox state
            use_gpu = False
            if hasattr(self, 'use_gpu_checkbox'):
                use_gpu = self.use_gpu_checkbox.isChecked()
            
            if use_gpu:
                # Try to select first GPU if available, otherwise CPU
                gpu_found = False
                for i in range(self.gpu_device_combo.count()):
                    data = self.gpu_device_combo.itemData(i)
                    if data and isinstance(data, str):
                        if data.startswith("cuda:") or data == "mps":
                            self.gpu_device_combo.setCurrentIndex(i)
                            gpu_found = True
                            break
                if not gpu_found:
                    # No GPU found, select CPU
                    for i in range(self.gpu_device_combo.count()):
                        data = self.gpu_device_combo.itemData(i)
                        if data == "cpu":
                            self.gpu_device_combo.setCurrentIndex(i)
                            break
            else:
                # Select CPU if GPU checkbox is unchecked
                for i in range(self.gpu_device_combo.count()):
                    data = self.gpu_device_combo.itemData(i)
                    if data == "cpu":
                        self.gpu_device_combo.setCurrentIndex(i)
                        break
            
            # Update GPU checkbox state based on availability
            if hasattr(self, 'use_gpu_checkbox'):
                gpu_available = is_gpu_available()
                if not gpu_available and self.use_gpu_checkbox.isChecked():
                    # GPU was requested but not available, uncheck and warn
                    self.use_gpu_checkbox.setChecked(False)
                    self.gpu_device_combo.setEnabled(False)
                    print("Warning: GPU requested but not available. Falling back to CPU.")
                        
        except Exception as e:
            # Error detecting devices
            self.gpu_device_combo.addItem("CPU (Error detecting devices)", "cpu")
            print(f"Error detecting GPU devices: {e}")
            import traceback
            traceback.print_exc()
        
        # Provide helpful diagnostic messages
        try:
            import torch
            import sys
            if not torch.cuda.is_available():
                if hasattr(torch.version, 'cuda') and torch.version.cuda is None:
                    print("\n" + "="*60)
                    print("WARNING: PyTorch is installed as CPU-only version.")
                    print(f"Python: {sys.executable}")
                    print(f"PyTorch: {torch.__version__}")
                    print("CUDA support is not available in this PyTorch build.")
                    print("\nTo enable GPU support:")
                    print("1. Make sure you're using the venv_gpu environment:")
                    print("   venv_gpu\\Scripts\\activate")
                    print("2. If using Python 3.14: PyTorch CUDA builds are not yet available.")
                    print("   Use Python 3.12 or 3.13 which have CUDA builds.")
                    print("3. If using Python 3.12/3.13, reinstall with:")
                    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                    print("="*60 + "\n")
                else:
                    # CUDA is compiled but not available (driver issue)
                    print(f"\nWARNING: PyTorch has CUDA {torch.version.cuda} compiled but CUDA is not available.")
                    print("This may be a driver issue. Check your NVIDIA drivers.\n")
        except ImportError:
            pass

    def on_gpu_checkbox_toggled(self, checked: bool):
        """Handle GPU checkbox toggle - enable/disable device dropdown and update selection."""
        if not hasattr(self, 'gpu_device_combo'):
            return  # Widget not created yet
        
        self.gpu_device_combo.setEnabled(checked)
        
        if not checked:
            # If unchecked, select CPU
            for i in range(self.gpu_device_combo.count()):
                data = self.gpu_device_combo.itemData(i)
                if data == "cpu":
                    self.gpu_device_combo.setCurrentIndex(i)
                    break
        else:
            # If checked, try to select first GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    # Select first CUDA device
                    for i in range(self.gpu_device_combo.count()):
                        data = self.gpu_device_combo.itemData(i)
                        if data and isinstance(data, str) and data.startswith("cuda:"):
                            self.gpu_device_combo.setCurrentIndex(i)
                            break
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # Select MPS if available
                    for i in range(self.gpu_device_combo.count()):
                        data = self.gpu_device_combo.itemData(i)
                        if data == "mps":
                            self.gpu_device_combo.setCurrentIndex(i)
                            break
            except Exception:
                pass

    def get_selected_device(self) -> str:
        """Get the selected device string from the dropdown."""
        if not hasattr(self, 'gpu_device_combo'):
            return "cpu"  # Default to CPU if widget not created
        
        device_data = self.gpu_device_combo.currentData()
        if device_data and isinstance(device_data, str):
            return device_data
        # Fallback to current text or CPU
        current_text = self.gpu_device_combo.currentText()
        if "CUDA" in current_text:
            # Extract CUDA device from text like "CUDA:0 (Device Name)"
            import re
            match = re.search(r'CUDA:(\d+)', current_text)
            if match:
                return f"cuda:{match.group(1)}"
        return "cpu"
