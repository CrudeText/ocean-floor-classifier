"""Training plot widget for real-time metrics visualization."""

from typing import Optional

import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QWidget

from ofc.core import TrainingHistory


class TrainingPlotWidget(QWidget):
    """Widget for displaying real-time training metrics using PyQtGraph."""

    def __init__(self, parent=None):
        """Initialize the training plot widget."""
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Loss plot (left)
        self.loss_plot = pg.PlotWidget(title="Loss")
        self.loss_plot.setLabel("left", "Loss")
        self.loss_plot.setLabel("bottom", "Epoch")
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_plot.addLegend()
        layout.addWidget(self.loss_plot)

        # Accuracy plot (right)
        self.accuracy_plot = pg.PlotWidget(title="Accuracy")
        self.accuracy_plot.setLabel("left", "Accuracy")
        self.accuracy_plot.setLabel("bottom", "Epoch")
        self.accuracy_plot.showGrid(x=True, y=True, alpha=0.3)
        self.accuracy_plot.addLegend()
        self.accuracy_plot.setYRange(0, 1)  # Accuracy is 0-1 (or 0-100%)
        layout.addWidget(self.accuracy_plot)

        # Initialize plot lines
        self.train_loss_line = self.loss_plot.plot(
            [], [], pen=pg.mkPen(color="b", width=2), name="Train"
        )
        self.val_loss_line = self.loss_plot.plot(
            [], [], pen=pg.mkPen(color="r", width=2), name="Validation"
        )

        self.train_acc_line = self.accuracy_plot.plot(
            [], [], pen=pg.mkPen(color="b", width=2), name="Train"
        )
        self.val_acc_line = self.accuracy_plot.plot(
            [], [], pen=pg.mkPen(color="r", width=2), name="Validation"
        )

        # Store data
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_accuracies: list[float] = []

        self.max_epochs: Optional[int] = None

    def set_max_epochs(self, max_epochs: int):
        """Set the maximum number of epochs for x-axis range."""
        self.max_epochs = max_epochs
        if max_epochs > 0:
            self.loss_plot.setXRange(0, max_epochs, padding=0.1)
            self.accuracy_plot.setXRange(0, max_epochs, padding=0.1)

    def update_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
    ):
        """
        Update plots with new metrics for an epoch.

        Args:
            epoch: Epoch number (1-indexed)
            train_loss: Training loss
            val_loss: Validation loss (optional)
            train_acc: Training accuracy (optional, 0-1 range)
            val_acc: Validation accuracy (optional, 0-1 range)
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)

        if val_loss is not None:
            self.val_losses.append(val_loss)
        else:
            # Pad with last value or None
            if self.val_losses:
                self.val_losses.append(self.val_losses[-1])
            else:
                self.val_losses.append(None)

        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        else:
            if self.train_accuracies:
                self.train_accuracies.append(self.train_accuracies[-1])
            else:
                self.train_accuracies.append(None)

        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        else:
            if self.val_accuracies:
                self.val_accuracies.append(self.val_accuracies[-1])
            else:
                self.val_accuracies.append(None)

        # Update loss plot
        self.train_loss_line.setData(self.epochs, self.train_losses)

        # Filter out None values for validation loss
        val_loss_data = [
            (e, v) for e, v in zip(self.epochs, self.val_losses) if v is not None
        ]
        if val_loss_data:
            val_epochs, val_losses = zip(*val_loss_data)
            self.val_loss_line.setData(list(val_epochs), list(val_losses))

        # Update accuracy plot
        train_acc_data = [
            (e, a) for e, a in zip(self.epochs, self.train_accuracies) if a is not None
        ]
        if train_acc_data:
            train_epochs, train_accs = zip(*train_acc_data)
            self.train_acc_line.setData(list(train_epochs), list(train_accs))

        val_acc_data = [
            (e, a) for e, a in zip(self.epochs, self.val_accuracies) if a is not None
        ]
        if val_acc_data:
            val_epochs, val_accs = zip(*val_acc_data)
            self.val_acc_line.setData(list(val_epochs), list(val_accs))

        # Auto-scale y-axis
        if self.train_losses:
            max_loss = max(self.train_losses)
            if self.val_losses:
                max_loss = max(max_loss, max(v for v in self.val_losses if v is not None))
            self.loss_plot.setYRange(0, max_loss * 1.1, padding=0.1)

    def load_history(self, history: TrainingHistory):
        """Load and display a complete training history."""
        self.clear()
        self.epochs = history.epochs.copy()
        self.train_losses = history.train_loss.copy()
        self.val_losses = history.val_loss.copy() if history.val_loss else []
        self.train_accuracies = history.train_accuracy.copy() if history.train_accuracy else []
        self.val_accuracies = history.val_accuracy.copy() if history.val_accuracy else []

        # Update plots
        self.train_loss_line.setData(self.epochs, self.train_losses)

        if self.val_losses:
            self.val_loss_line.setData(self.epochs, self.val_losses)

        if self.train_accuracies:
            self.train_acc_line.setData(self.epochs, self.train_accuracies)

        if self.val_accuracies:
            self.val_acc_line.setData(self.epochs, self.val_accuracies)

        # Set x-axis range
        if self.epochs:
            max_epoch = max(self.epochs)
            self.loss_plot.setXRange(0, max_epoch, padding=0.1)
            self.accuracy_plot.setXRange(0, max_epoch, padding=0.1)

        # Auto-scale y-axis
        if self.train_losses:
            max_loss = max(self.train_losses)
            if self.val_losses:
                max_loss = max(max_loss, max(self.val_losses))
            self.loss_plot.setYRange(0, max_loss * 1.1, padding=0.1)

    def clear(self):
        """Clear all plots and reset data."""
        self.epochs.clear()
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        self.train_loss_line.setData([], [])
        self.val_loss_line.setData([], [])
        self.train_acc_line.setData([], [])
        self.val_acc_line.setData([], [])

        self.max_epochs = None
