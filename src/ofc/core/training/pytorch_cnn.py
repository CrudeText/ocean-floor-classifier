"""PyTorch CNN training implementation."""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ArchitectureConfig, TrainingConfig
from .device import get_device


class ConfigurableCNN(nn.Module):
    """
    Configurable Convolutional Neural Network built from ArchitectureConfig.

    Dynamically constructs a PyTorch model based on layer configurations.
    """

    def __init__(self, config: ArchitectureConfig):
        """
        Initialize model from architecture configuration.

        Args:
            config: ArchitectureConfig defining the model structure

        Raises:
            ValueError: If architecture is invalid
        """
        super().__init__()
        config.validate()

        self.config = config
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.activations = []
        self.layer_types = []  # Track layer types for forward pass

        # Track dimensions for automatic layer sizing
        # Start with input size
        channels = 3  # RGB input
        height, width = config.input_size
        flattened = False

        # Build layers
        for layer_cfg in config.layers:
            layer_type = layer_cfg.layer_type
            
            if layer_type == "linear":
                # Linear layers require flattening first
                if not flattened:
                    # Calculate flattened size
                    feature_size = channels * height * width
                    flattened = True
                else:
                    # Subsequent linear layers use previous output size
                    feature_size = self.linear_layers[-1].out_features if self.linear_layers else channels * height * width
                
                out_features = layer_cfg.params["out_features"]
                layer = nn.Linear(feature_size, out_features)
                self.linear_layers.append(layer)
                self.layer_types.append("linear")
            else:
                # Convolutional/pooling layers
                layer, channels, height, width = self._build_layer(
                    layer_cfg, channels, height, width
                )
                self.conv_layers.append(layer)
                self.layer_types.append(layer_type)
            
            self.activations.append(layer_cfg.activation)

        # Final classification layer
        # If we have linear layers, use the last one's output size
        # Otherwise, flatten the conv output
        if self.linear_layers:
            classifier_input_size = self.linear_layers[-1].out_features
        else:
            classifier_input_size = channels * height * width

        self.classifier = nn.Linear(classifier_input_size, config.num_classes)

    def _build_layer(
        self, layer_cfg, in_channels: int, height: int, width: int
    ) -> tuple[nn.Module, int, int, int]:
        """
        Build a single layer and return updated dimensions.

        Args:
            layer_cfg: LayerConfig for this layer
            in_channels: Current number of input channels
            height: Current feature map height
            width: Current feature map width

        Returns:
            Tuple of (layer_module, out_channels, new_height, new_width)
        """
        layer_type = layer_cfg.layer_type
        params = layer_cfg.params

        if layer_type == "conv2d":
            out_channels = params["out_channels"]
            kernel_size = params.get("kernel_size", 3)
            stride = params.get("stride", 1)
            padding = params.get("padding", 1)

            layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

            # Update dimensions
            # Formula: out = (in + 2*padding - kernel_size) / stride + 1
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1

            return layer, out_channels, height, width

        elif layer_type == "maxpool2d":
            kernel_size = params.get("kernel_size", 2)
            stride = params.get("stride", 2)
            padding = params.get("padding", 0)

            layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

            # Update dimensions
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1

            return layer, in_channels, height, width

        elif layer_type == "avgpool2d":
            kernel_size = params.get("kernel_size", 2)
            stride = params.get("stride", 2)
            padding = params.get("padding", 0)

            layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

            # Update dimensions
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1

            return layer, in_channels, height, width

        elif layer_type == "batchnorm2d":
            layer = nn.BatchNorm2d(in_channels)
            return layer, in_channels, height, width

        elif layer_type == "dropout":
            # Dropout doesn't change dimensions, but we handle it in forward
            # For 2D dropout, we use Dropout2d
            p = params.get("p", 0.5)
            layer = nn.Dropout2d(p=p)
            return layer, in_channels, height, width

        # Note: Linear layers are handled separately in __init__
        # This method only handles conv/pool layers

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Process through all layers
        conv_idx = 0
        linear_idx = 0
        flattened = False

        for i, (layer_type, activation) in enumerate(zip(self.layer_types, self.activations)):
            if layer_type == "linear":
                # Flatten before first linear layer
                if not flattened:
                    x = x.view(x.size(0), -1)
                    flattened = True
                
                # Apply linear layer
                x = self.linear_layers[linear_idx](x)
                linear_idx += 1
            else:
                # Apply conv/pool layer
                x = self.conv_layers[conv_idx](x)
                conv_idx += 1

            # Apply activation (use in-place operations where possible for efficiency)
            if activation == "relu":
                x = F.relu(x, inplace=True)  # In-place ReLU for better performance
            elif activation == "sigmoid":
                x = torch.sigmoid(x)
            elif activation == "tanh":
                x = torch.tanh(x)
            # softmax is applied at the end, not here

        # Flatten if not already flattened (for final classifier)
        if not flattened:
            x = x.view(x.size(0), -1)

        # Classification layer
        x = self.classifier(x)

        return x


class PytorchTrainer:
    """
    PyTorch trainer for configurable CNN models.

    Handles training loop, validation, checkpointing, and GPU support.
    """

    def __init__(self, config: TrainingConfig, project: "OceanProject"):
        """
        Initialize trainer.

        Args:
            config: TrainingConfig with all training parameters
            project: OceanProject instance for accessing data
        """
        from ..project import OceanProject

        config.validate()
        self.config = config
        self.project = project

        # Determine device
        self.device = self._get_device()

        # cuDNN benchmarking disabled - it causes first batch to take 3-4 minutes and can cause CUDA memory errors
        # Batches will be slightly slower but more consistent and stable
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            # Enable TensorFloat32 (TF32) for faster matrix multiplication on Ampere+ GPUs
            # TF32 provides ~10x speedup for matrix ops with minimal accuracy loss
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')  # Use TF32 for matmul
            
            print(f"[Trainer] CUDA optimizations: cudnn.benchmark=False (disabled for stability)")
            print(f"[Trainer] TensorFloat32 (TF32) enabled for faster matrix operations")

        # Build model
        self.model = ConfigurableCNN(config.architecture).to(self.device)
        
        # Try to compile model for better performance (PyTorch 2.0+)
        # Note: On Windows, Triton (required for inductor backend) is not available
        # So we skip compilation on Windows
        import sys
        if sys.platform == 'win32':
            print(f"[Trainer] Model compilation skipped (Triton not available on Windows)")
        else:
            try:
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    print(f"[Trainer] Model compiled with torch.compile for better performance")
            except Exception as e:
                print(f"[Trainer] Model compilation not available or failed: {e}")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Model: {total_params:,} total parameters, {trainable_params:,} trainable")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = []

    def _get_device(self) -> torch.device:
        """
        Get the appropriate device (CPU or GPU).

        Returns:
            torch.device instance
        """
        # Get device string from config if available, otherwise use use_gpu flag
        device_string = getattr(self.config, 'device', None)
        return get_device(self.config.use_gpu, device_string)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        progress_callback: Optional[callable] = None,
    ) -> list[dict]:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            progress_callback: Optional callback(epoch, metrics_dict) called each epoch

        Returns:
            List of metrics dictionaries, one per epoch
        """
        # Setup optimizer
        optimizer = self._create_optimizer()

        # Setup loss function
        criterion = self._create_loss_function()

        # Setup learning rate scheduler (optional)
        scheduler = None
        if "scheduler" in self.config.optimizer_params:
            scheduler_type = self.config.optimizer_params["scheduler"]
            if scheduler_type == "step":
                step_size = self.config.optimizer_params.get("step_size", 10)
                gamma = self.config.optimizer_params.get("gamma", 0.1)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=step_size, gamma=gamma
                )

        # Early stopping
        early_stopping_patience = None
        early_stopping_min_delta = 0.0
        if self.config.early_stopping:
            early_stopping_patience = self.config.early_stopping["patience"]
            early_stopping_min_delta = self.config.early_stopping.get("min_delta", 0.0)
        patience_counter = 0

        import time
        from datetime import datetime
        
        def log_progress(message: str):
            """Log progress with timestamp."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [Trainer] {message}")
        
        history = []
        best_val_loss = float("inf")
        
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader) if val_loader else 0
        log_progress(f"Training loop: {num_train_batches} train batches, {num_val_batches} val batches per epoch")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            log_progress(f"Epoch {epoch + 1}/{self.config.num_epochs} starting...")

            # Training phase
            train_start = time.time()
            log_progress(f"  Training phase...")
            train_metrics = self._train_epoch(train_loader, optimizer, criterion)
            train_time = time.time() - train_start
            log_progress(f"  Training completed in {train_time:.2f} seconds ({train_time/num_train_batches*1000:.1f} ms/batch)")

            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_start = time.time()
                log_progress(f"  Validation phase...")
                val_metrics = self._validate_epoch(val_loader, criterion)
                val_time = time.time() - val_start
                log_progress(f"  Validation completed in {val_time:.2f} seconds ({val_time/num_val_batches*1000:.1f} ms/batch)")
            else:
                log_progress(f"  No validation (val_loader is None)")

            # Update learning rate
            if scheduler is not None:
                lr_before = optimizer.param_groups[0]['lr']
                scheduler.step()
                lr_after = optimizer.param_groups[0]['lr']
                train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
                if lr_before != lr_after:
                    log_progress(f"  Learning rate changed: {lr_before:.6f} -> {lr_after:.6f}")

            # Combine metrics
            metrics = {
                "epoch": epoch,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(metrics)
            
            epoch_time = time.time() - epoch_start
            log_progress(f"  Epoch {epoch + 1} total time: {epoch_time:.2f} seconds")
            log_progress(f"  GPU state after epoch: {get_gpu_memory_info()}")

            # Progress callback
            if progress_callback is not None:
                progress_callback(epoch, metrics)

            # Early stopping check
            if val_loader is not None and early_stopping_patience is not None:
                val_loss = val_metrics.get("loss", float("inf"))
                if val_loss < self.best_val_loss - early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        self.training_history = history
        return history

    def _train_epoch(
        self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion
    ) -> dict:
        """Train for one epoch."""
        import time
        from datetime import datetime
        
        def log_progress(message: str):
            """Log progress with timestamp."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [TrainEpoch] {message}")
        
        def get_gpu_memory_info() -> str:
            """Get GPU memory usage info as a string."""
            if self.device.type != 'cuda':
                return "N/A"
            try:
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(self.device) / (1024**3)  # GB
                max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)  # GB
                return f"GPU: {allocated:.2f}GB/{reserved:.2f}GB (peak: {max_allocated:.2f}GB)"
            except Exception:
                return "N/A"
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(train_loader)
        log_interval = max(1, num_batches // 10)  # Log every 10% of batches
        
        epoch_start_time = time.time()
        prev_batch_end_time = epoch_start_time
        
        # Log initial GPU memory
        log_progress(f"Starting epoch - {get_gpu_memory_info()}")
        log_progress(f"Will log batches: 1-5 (first 5), then every {log_interval} batches")

        for batch_idx, (data, target) in enumerate(train_loader):
            # Measure time since last batch ended (includes data loading overhead)
            time_since_last_batch = time.time() - prev_batch_end_time
            
            # Log when we start each of the first few batches (to catch data loading delays)
            if batch_idx < 5:
                log_progress(f"  Starting batch {batch_idx + 1}/{num_batches} (time since last: {time_since_last_batch*1000:.1f}ms)")
            
            batch_start = time.time()
            
            # Data transfer to device
            data_transfer_start = time.time()
            data, target = data.to(self.device), target.to(self.device)
            data_transfer_time = time.time() - data_transfer_start

            # Forward pass
            forward_start = time.time()
            output = self.model(data)
            
            # Synchronize after model forward (before loss calculation)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            forward_model_time = time.time() - forward_start
            
            # Loss calculation
            loss_start = time.time()
            loss = criterion(output, target)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            loss_time = time.time() - loss_start
            forward_time = forward_model_time + loss_time

            # Backward pass
            backward_start = time.time()
            optimizer.zero_grad()
            
            # Backward pass timing
            backward_compute_start = time.time()
            loss.backward()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            backward_compute_time = time.time() - backward_compute_start
            
            # Warn if backward pass is unusually slow (likely cuDNN autotuning on first batch)
            if batch_idx == 0 and backward_compute_time > 10.0:
                log_progress(f"    NOTE: First batch backward pass took {backward_compute_time:.1f}s - this is cuDNN autotuning, subsequent batches will be ~3s")
            
            # Optimizer step timing
            optimizer_step_start = time.time()
            optimizer.step()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            optimizer_step_time = time.time() - optimizer_step_start
            
            backward_time = time.time() - backward_start

            # Metrics
            # Note: GPU operations are asynchronous, so we need to synchronize before measuring
            metrics_start = time.time()
            
            # Synchronize GPU before getting metrics (ensures all operations are complete)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Now get metrics (should be fast after sync)
            batch_loss = loss.item()
            total_loss += batch_loss
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            metrics_time = time.time() - metrics_start
            
            batch_time = time.time() - batch_start
            prev_batch_end_time = time.time()
            
            # Log progress for first 5 batches (to catch warmup issues), then every Nth batch
            should_log = (
                (batch_idx + 1) <= 5 or  # Always log first 5 batches
                (batch_idx + 1) % log_interval == 0 or  # Then every Nth batch
                (batch_idx + 1) == num_batches  # And the last batch
            )
            
            if should_log:
                data_load_time = max(0, time_since_last_batch - batch_time)  # Approximate data loading time
                gpu_mem = get_gpu_memory_info()
                log_progress(
                    f"  Batch {batch_idx + 1}/{num_batches}: loss={batch_loss:.4f}, "
                    f"batch={batch_time*1000:.1f}ms, "
                    f"since_last={time_since_last_batch*1000:.1f}ms "
                    f"(data_load≈{data_load_time*1000:.1f}ms, "
                    f"transfer={data_transfer_time*1000:.1f}ms, "
                    f"forward={forward_time*1000:.1f}ms[model={forward_model_time*1000:.1f}ms+loss={loss_time*1000:.1f}ms], "
                    f"backward={backward_time*1000:.1f}ms[grad={backward_compute_time*1000:.1f}ms+step={optimizer_step_time*1000:.1f}ms], "
                    f"metrics={metrics_time*1000:.1f}ms) "
                    f"{gpu_mem}"
                )

        return {
            "loss": total_loss / len(train_loader),
            "accuracy": correct / total if total > 0 else 0.0,
        }

    def _validate_epoch(self, val_loader: DataLoader, criterion) -> dict:
        """Validate for one epoch."""
        import time
        from datetime import datetime
        
        def log_progress(message: str):
            """Log progress with timestamp."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [ValEpoch] {message}")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(val_loader)
        log_interval = max(1, num_batches // 10)  # Log every 10% of batches
        
        val_start_time = time.time()
        prev_batch_end_time = val_start_time

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                # Measure time since last batch ended (includes data loading overhead)
                time_since_last_batch = time.time() - prev_batch_end_time
                batch_start = time.time()
                
                # Data transfer to device
                data_transfer_start = time.time()
                data, target = data.to(self.device), target.to(self.device)
                data_transfer_time = time.time() - data_transfer_start

                # Forward pass
                forward_start = time.time()
                output = self.model(data)
                loss = criterion(output, target)
                forward_time = time.time() - forward_start

            # Metrics
            # Note: GPU operations are asynchronous, so we need to synchronize before measuring
            metrics_start = time.time()
            
            # Synchronize GPU before getting metrics (ensures all operations are complete)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Now get metrics (should be fast after sync)
            batch_loss = loss.item()
            total_loss += batch_loss
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            metrics_time = time.time() - metrics_start
            
            batch_time = time.time() - batch_start
            prev_batch_end_time = time.time()
            
            # Log progress for every Nth batch with detailed timing
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                data_load_time = max(0, time_since_last_batch - batch_time)  # Approximate data loading time
                gpu_mem = get_gpu_memory_info()
                log_progress(
                    f"  Batch {batch_idx + 1}/{num_batches}: loss={batch_loss:.4f}, "
                    f"batch={batch_time*1000:.1f}ms, "
                    f"since_last={time_since_last_batch*1000:.1f}ms "
                    f"(data_load≈{data_load_time*1000:.1f}ms, "
                    f"transfer={data_transfer_time*1000:.1f}ms, "
                    f"forward={forward_time*1000:.1f}ms, "
                    f"metrics={metrics_time*1000:.1f}ms) "
                    f"{gpu_mem}"
                )

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total if total > 0 else 0.0,
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        lr = self.config.learning_rate

        if self.config.optimizer == "adam":
            weight_decay = self.config.optimizer_params.get("weight_decay", 0.0)
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif self.config.optimizer == "sgd":
            momentum = self.config.optimizer_params.get("momentum", 0.9)
            weight_decay = self.config.optimizer_params.get("weight_decay", 0.0)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif self.config.optimizer == "rmsprop":
            weight_decay = self.config.optimizer_params.get("weight_decay", 0.0)
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        elif self.config.optimizer == "adamw":
            weight_decay = self.config.optimizer_params.get("weight_decay", 0.01)
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_loss_function(self):
        """Create loss function based on config."""
        if self.config.loss_function == "cross_entropy":
            # Handle class weights if specified
            weight = None
            if self.config.class_weights:
                # Class weights should be provided via dataset
                # For now, we'll use None and handle it in the training loop
                pass
            return nn.CrossEntropyLoss(weight=weight)
        elif self.config.loss_function == "nll_loss":
            return nn.NLLLoss()
        elif self.config.loss_function == "focal_loss":
            # Focal loss not in standard PyTorch, would need custom implementation
            # For now, fall back to cross entropy
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")

    def validate(self, val_loader: DataLoader) -> dict:
        """
        Validate the model on validation data.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary with validation metrics
        """
        criterion = self._create_loss_function()
        return self._validate_epoch(val_loader, criterion)

    def save_checkpoint(
        self, path: Path, epoch: int, metrics: dict, is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint (.pth file)
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            is_best: Whether this is the best model so far
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "metrics": metrics,
            "is_best": is_best,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> tuple[int, dict]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Tuple of (epoch, metrics_dict)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Return epoch and metrics
        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})

        return epoch, metrics
