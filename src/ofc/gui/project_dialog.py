"""Project creation dialog."""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ofc.core import GridSpec, get_image_size, list_images, read_image_pil
from ofc.gui.utils_qt import pil_to_qpixmap


class ProjectCreationDialog(QDialog):
    """Dialog for configuring new project parameters."""

    def __init__(self, parent=None, default_name: str = ""):
        """Initialize project creation dialog."""
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.setMinimumWidth(700)  # Reduced width
        self.setMinimumHeight(700)
        self.resize(750, 800)  # Reduced default width
        self.project_folder: Optional[Path] = None
        # Set default raw images folder path for testing
        default_raw_path = Path(
            r"D:\A - Project Data\OceanFloorClassifier\Usable Data\JPG"
        )
        if default_raw_path.exists():
            self.raw_images_folder: Optional[Path] = default_raw_path
        else:
            self.raw_images_folder: Optional[Path] = None
        self.loaded_images: list[Path] = []
        self.current_image_idx: int = 0
        self.loaded_image_pixmap: Optional[QPixmap] = None
        self.loaded_image_size: tuple[int, int] = (0, 0)
        # Border values
        self.border_top: int = 0
        self.border_bottom: int = 0
        self.border_left: int = 0
        self.border_right: int = 0
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Form layout for parameters
        form_layout = QFormLayout()
        
        # Project name
        self.name_input = QLineEdit()
        self.name_input.setText(default_name)
        form_layout.addRow("Project Name:", self.name_input)
        
        # Grid Parameters Group Box
        grid_params_group = QGroupBox("Grid Parameters")
        grid_params_layout = QHBoxLayout()
        grid_params_group.setLayout(grid_params_layout)
        
        # Left side: Borders
        borders_container = QWidget()
        borders_layout = QVBoxLayout()
        borders_container.setLayout(borders_layout)
        
        borders_title = QLabel("Borders (grid offset from edges):")
        borders_layout.addWidget(borders_title)
        
        # Top and Bottom with symmetry
        top_bottom_layout = QHBoxLayout()
        
        top_layout = QVBoxLayout()
        top_label = QLabel("Top (px):")
        self.border_top_spinbox = QSpinBox()
        self.border_top_spinbox.setMinimum(0)
        self.border_top_spinbox.setMaximum(10000)
        self.border_top_spinbox.setValue(0)
        self.border_top_spinbox.setSingleStep(10)  # Larger step for cleaner interface
        self.border_top_spinbox.valueChanged.connect(self.on_border_top_changed)
        top_layout.addWidget(top_label)
        top_layout.addWidget(self.border_top_spinbox)
        top_bottom_layout.addLayout(top_layout)
        
        bottom_layout = QVBoxLayout()
        bottom_label = QLabel("Bottom (px):")
        self.border_bottom_spinbox = QSpinBox()
        self.border_bottom_spinbox.setMinimum(0)
        self.border_bottom_spinbox.setMaximum(10000)
        self.border_bottom_spinbox.setValue(0)
        self.border_bottom_spinbox.setSingleStep(10)  # Larger step for cleaner interface
        self.border_bottom_spinbox.valueChanged.connect(self.on_border_bottom_changed)
        bottom_layout.addWidget(bottom_label)
        bottom_layout.addWidget(self.border_bottom_spinbox)
        top_bottom_layout.addLayout(bottom_layout)
        
        self.symmetry_v_checkbox = QCheckBox("Symmetry")
        self.symmetry_v_checkbox.setChecked(True)
        top_bottom_layout.addWidget(self.symmetry_v_checkbox)
        top_bottom_layout.addStretch()
        
        borders_layout.addLayout(top_bottom_layout)
        
        # Left and Right with symmetry
        left_right_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        left_label = QLabel("Left (px):")
        self.border_left_spinbox = QSpinBox()
        self.border_left_spinbox.setMinimum(0)
        self.border_left_spinbox.setMaximum(10000)
        self.border_left_spinbox.setValue(0)
        self.border_left_spinbox.setSingleStep(10)  # Larger step for cleaner interface
        self.border_left_spinbox.valueChanged.connect(self.on_border_left_changed)
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.border_left_spinbox)
        left_right_layout.addLayout(left_layout)
        
        right_layout = QVBoxLayout()
        right_label = QLabel("Right (px):")
        self.border_right_spinbox = QSpinBox()
        self.border_right_spinbox.setMinimum(0)
        self.border_right_spinbox.setMaximum(10000)
        self.border_right_spinbox.setValue(0)
        self.border_right_spinbox.setSingleStep(10)  # Larger step for cleaner interface
        self.border_right_spinbox.valueChanged.connect(self.on_border_right_changed)
        right_layout.addWidget(right_label)
        right_layout.addWidget(self.border_right_spinbox)
        left_right_layout.addLayout(right_layout)
        
        self.symmetry_h_checkbox = QCheckBox("Symmetry")
        self.symmetry_h_checkbox.setChecked(True)
        left_right_layout.addWidget(self.symmetry_h_checkbox)
        left_right_layout.addStretch()
        
        borders_layout.addLayout(left_right_layout)
        grid_params_layout.addWidget(borders_container)
        
        # Right side: Tile Size and Stride
        right_side_container = QWidget()
        right_side_layout = QVBoxLayout()
        right_side_container.setLayout(right_side_layout)
        
        # Tile size (width and height on same row)
        tile_size_layout = QHBoxLayout()
        
        tile_w_layout = QVBoxLayout()
        tile_w_label = QLabel("Tile Width (px):")
        self.tile_w_spinbox = QSpinBox()
        self.tile_w_spinbox.setMinimum(32)
        self.tile_w_spinbox.setMaximum(2048)
        self.tile_w_spinbox.setValue(256)
        self.tile_w_spinbox.setSingleStep(32)
        self.tile_w_spinbox.valueChanged.connect(self.update_preview)
        tile_w_layout.addWidget(tile_w_label)
        tile_w_layout.addWidget(self.tile_w_spinbox)
        tile_size_layout.addLayout(tile_w_layout)
        
        tile_h_layout = QVBoxLayout()
        tile_h_label = QLabel("Tile Height (px):")
        self.tile_h_spinbox = QSpinBox()
        self.tile_h_spinbox.setMinimum(32)
        self.tile_h_spinbox.setMaximum(2048)
        self.tile_h_spinbox.setValue(256)
        self.tile_h_spinbox.setSingleStep(32)
        self.tile_h_spinbox.valueChanged.connect(self.update_preview)
        tile_h_layout.addWidget(tile_h_label)
        tile_h_layout.addWidget(self.tile_h_spinbox)
        tile_size_layout.addLayout(tile_h_layout)
        
        tile_size_widget = QWidget()
        tile_size_widget.setLayout(tile_size_layout)
        right_side_layout.addWidget(tile_size_widget)
        
        # Stride (X and Y on same row)
        stride_layout = QHBoxLayout()
        
        stride_x_layout = QVBoxLayout()
        stride_x_label = QLabel("Stride X (px, 0 = width):")
        self.stride_x_spinbox = QSpinBox()
        self.stride_x_spinbox.setMinimum(0)
        self.stride_x_spinbox.setMaximum(2048)
        self.stride_x_spinbox.setValue(0)  # 0 means use tile width
        self.stride_x_spinbox.setSingleStep(32)
        stride_x_layout.addWidget(stride_x_label)
        stride_x_layout.addWidget(self.stride_x_spinbox)
        stride_layout.addLayout(stride_x_layout)
        
        stride_y_layout = QVBoxLayout()
        stride_y_label = QLabel("Stride Y (px, 0 = height):")
        self.stride_y_spinbox = QSpinBox()
        self.stride_y_spinbox.setMinimum(0)
        self.stride_y_spinbox.setMaximum(2048)
        self.stride_y_spinbox.setValue(0)  # 0 means use tile height
        self.stride_y_spinbox.setSingleStep(32)
        stride_y_layout.addWidget(stride_y_label)
        stride_y_layout.addWidget(self.stride_y_spinbox)
        stride_layout.addLayout(stride_y_layout)
        
        # Connect stride changes to update preview (after spinboxes are created)
        self.stride_x_spinbox.valueChanged.connect(self.update_preview)
        self.stride_y_spinbox.valueChanged.connect(self.update_preview)
        
        stride_widget = QWidget()
        stride_widget.setLayout(stride_layout)
        right_side_layout.addWidget(stride_widget)
        
        grid_params_layout.addWidget(right_side_container)
        
        form_layout.addRow(grid_params_group)
        
        # Tile Preview Group Box
        preview_group = QGroupBox("Tile Preview")
        preview_container_layout = QVBoxLayout()
        preview_group.setLayout(preview_container_layout)
        
        # Load image button and navigation (inside group box)
        preview_header_layout = QHBoxLayout()
        preview_header_layout.addStretch()
        
        load_image_btn = QPushButton("Load Image...")
        load_image_btn.clicked.connect(self.load_image_from_folder)
        preview_header_layout.addWidget(load_image_btn)
        
        self.prev_image_btn = QPushButton("◀")
        self.prev_image_btn.setEnabled(False)
        self.prev_image_btn.clicked.connect(self.prev_image)
        self.prev_image_btn.setMaximumWidth(30)
        preview_header_layout.addWidget(self.prev_image_btn)
        
        self.next_image_btn = QPushButton("▶")
        self.next_image_btn.setEnabled(False)
        self.next_image_btn.clicked.connect(self.next_image)
        self.next_image_btn.setMaximumWidth(30)
        preview_header_layout.addWidget(self.next_image_btn)
        
        preview_container_layout.addLayout(preview_header_layout)
        
        # Resolution selector with custom inputs
        resolution_layout = QHBoxLayout()
        resolution_label = QLabel("Image Resolution:")
        resolution_layout.addWidget(resolution_label)
        self.resolution_combo = QComboBox()
        # Classic resolutions
        self.resolutions = [
            ("1920×1080 (Full HD)", 1920, 1080),
            ("2560×1440 (2K)", 2560, 1440),
            ("3840×2160 (4K)", 3840, 2160),
            ("5120×2880 (5K)", 5120, 2880),
            ("7680×4320 (8K)", 7680, 4320),
            ("1024×768 (XGA)", 1024, 768),
            ("1280×720 (HD)", 1280, 720),
            ("1600×900 (HD+)", 1600, 900),
            ("2048×1536 (QXGA)", 2048, 1536),
            ("Custom", 0, 0),  # Special marker for custom
        ]
        for name, w, h in self.resolutions:
            self.resolution_combo.addItem(name, (w, h))
        self.resolution_combo.currentIndexChanged.connect(self.on_resolution_changed)
        resolution_layout.addWidget(self.resolution_combo)
        
        # Custom resolution inputs (disabled by default, shown when Custom is selected)
        custom_x_label = QLabel("X:")
        self.custom_x_input = QSpinBox()
        self.custom_x_input.setMinimum(1)
        self.custom_x_input.setMaximum(100000)
        self.custom_x_input.setValue(1920)
        self.custom_x_input.setEnabled(False)
        self.custom_x_input.valueChanged.connect(self.update_preview)
        resolution_layout.addWidget(custom_x_label)
        resolution_layout.addWidget(self.custom_x_input)
        
        custom_y_label = QLabel("Y:")
        self.custom_y_input = QSpinBox()
        self.custom_y_input.setMinimum(1)
        self.custom_y_input.setMaximum(100000)
        self.custom_y_input.setValue(1080)
        self.custom_y_input.setEnabled(False)
        self.custom_y_input.valueChanged.connect(self.update_preview)
        resolution_layout.addWidget(custom_y_label)
        resolution_layout.addWidget(self.custom_y_input)
        
        resolution_layout.addStretch()
        preview_container_layout.addLayout(resolution_layout)
        
        # Preview display with fixed size to prevent cropping
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(300, 225)
        self.preview_label.setMaximumSize(450, 337)
        self.preview_label.setFixedSize(450, 337)  # Force to max size (smaller)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # Prevent resizing
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_container_layout.addWidget(self.preview_label)
        
        form_layout.addRow(preview_group)
        
        # Project folder selection (underneath tile preview)
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Not selected")
        folder_layout.addWidget(self.folder_label)
        folder_layout.addStretch()
        choose_folder_btn = QPushButton("Choose Folder...")
        choose_folder_btn.clicked.connect(self.choose_folder)
        folder_layout.addWidget(choose_folder_btn)
        folder_widget = QWidget()
        folder_widget.setLayout(folder_layout)
        form_layout.addRow("Project Folder:", folder_widget)
        
        # Raw images folder selection
        raw_images_layout = QHBoxLayout()
        self.raw_images_folder_label = QLabel("Not selected")
        # Set label text based on initialized raw_images_folder
        if self.raw_images_folder:
            self.raw_images_folder_label.setText(str(self.raw_images_folder))
        raw_images_layout.addWidget(self.raw_images_folder_label)
        raw_images_layout.addStretch()
        choose_raw_images_btn = QPushButton("Choose Folder...")
        choose_raw_images_btn.clicked.connect(self.choose_raw_images_folder)
        raw_images_layout.addWidget(choose_raw_images_btn)
        raw_images_widget = QWidget()
        raw_images_widget.setLayout(raw_images_layout)
        form_layout.addRow("Raw Images Folder:", raw_images_widget)
        
        layout.addLayout(form_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # No default classes - start with empty list
        
        # Try to load images from main window if available
        self.try_load_images_from_main_window()
    
    def try_load_images_from_main_window(self):
        """Try to load images from the main window's label tab if available."""
        if self.parent():
            try:
                # Try to access main window's label tab
                main_window = self.parent()
                if hasattr(main_window, 'label_tab'):
                    label_tab = main_window.label_tab
                    if hasattr(label_tab, 'images_list') and label_tab.images_list.count() > 0:
                        # Get images from label tab
                        images = []
                        for i in range(label_tab.images_list.count()):
                            item = label_tab.images_list.item(i)
                            if item:
                                # Check if it's a preview folder image
                                if hasattr(label_tab, 'preview_folder') and label_tab.preview_folder:
                                    img_path_str = item.data(Qt.ItemDataRole.UserRole)
                                    if img_path_str:
                                        images.append(Path(img_path_str))
                                # Or project image
                                elif hasattr(label_tab, 'project') and label_tab.project:
                                    img_rel_path = item.text()
                                    images.append(label_tab.project.get_raw_image_path(img_rel_path))
                        
                        if images:
                            self.loaded_images = images
                            self.current_image_idx = 0
                            self.load_image_from_path(self.loaded_images[0])
            except Exception:
                # Silently fail if we can't access main window
                pass
    
    def load_image_from_folder(self):
        """Load an image from a folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Folder with Images", "", QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            try:
                image_paths = list_images(Path(folder))
                if image_paths:
                    self.loaded_images = image_paths
                    self.current_image_idx = 0
                    self.load_image_from_path(self.loaded_images[0])
                else:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "No Images", "No images found in the selected folder.")
            except Exception as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", f"Failed to load images: {str(e)}")
    
    def load_image_from_path(self, image_path: Path):
        """Load an image from a specific path."""
        try:
            img = read_image_pil(image_path)
            self.loaded_image_pixmap = pil_to_qpixmap(img)
            self.loaded_image_size = img.size  # (width, height)
            
            # Update resolution to match loaded image
            img_w, img_h = self.loaded_image_size
            self.update_resolution_display(img_w, img_h)
            
            self.update_navigation_buttons()
            self.update_preview()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")
    
    def update_resolution_display(self, img_w: int, img_h: int):
        """Update resolution dropdown and custom inputs to match detected resolution."""
        # Find matching resolution
        found_match = False
        for i, (name, w, h) in enumerate(self.resolutions):
            if w > 0 and w == img_w and h == img_h:
                self.resolution_combo.setCurrentIndex(i)
                self.custom_x_input.setEnabled(False)
                self.custom_y_input.setEnabled(False)
                found_match = True
                break
        
        if not found_match:
            # Set to custom and update values
            self.resolution_combo.setCurrentIndex(self.resolution_combo.count() - 1)  # Custom
            self.custom_x_input.setValue(img_w)
            self.custom_y_input.setValue(img_h)
            self.custom_x_input.setEnabled(True)
            self.custom_y_input.setEnabled(True)
    
    def prev_image(self):
        """Load previous image."""
        if self.loaded_images and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_image_from_path(self.loaded_images[self.current_image_idx])
    
    def next_image(self):
        """Load next image."""
        if self.loaded_images and self.current_image_idx < len(self.loaded_images) - 1:
            self.current_image_idx += 1
            self.load_image_from_path(self.loaded_images[self.current_image_idx])
    
    def update_navigation_buttons(self):
        """Update navigation button states."""
        has_images = len(self.loaded_images) > 0
        self.prev_image_btn.setEnabled(has_images and self.current_image_idx > 0)
        self.next_image_btn.setEnabled(has_images and self.current_image_idx < len(self.loaded_images) - 1)
    
    def on_resolution_changed(self):
        """Handle resolution combo box change."""
        resolution_data = self.resolution_combo.currentData()
        is_custom = resolution_data and resolution_data[0] == 0
        
        # Enable/disable custom inputs
        self.custom_x_input.setEnabled(is_custom)
        self.custom_y_input.setEnabled(is_custom)
        
        # If switching to custom and we have a loaded image, use its size
        if is_custom and self.loaded_image_size[0] > 0:
            self.custom_x_input.setValue(self.loaded_image_size[0])
            self.custom_y_input.setValue(self.loaded_image_size[1])
        elif not is_custom and resolution_data:
            # Update custom inputs to show the selected resolution (for reference)
            w, h = resolution_data
            self.custom_x_input.setValue(w)
            self.custom_y_input.setValue(h)
        
        self.update_preview()
    
    def on_border_top_changed(self):
        """Handle top border change."""
        self.border_top = self.border_top_spinbox.value()
        if self.symmetry_v_checkbox.isChecked():
            self.border_bottom_spinbox.setValue(self.border_top)
        self.update_preview()
    
    def on_border_bottom_changed(self):
        """Handle bottom border change."""
        self.border_bottom = self.border_bottom_spinbox.value()
        if self.symmetry_v_checkbox.isChecked():
            self.border_top_spinbox.setValue(self.border_bottom)
        self.update_preview()
    
    def on_border_left_changed(self):
        """Handle left border change."""
        self.border_left = self.border_left_spinbox.value()
        if self.symmetry_h_checkbox.isChecked():
            self.border_right_spinbox.setValue(self.border_left)
        self.update_preview()
    
    def on_border_right_changed(self):
        """Handle right border change."""
        self.border_right = self.border_right_spinbox.value()
        if self.symmetry_h_checkbox.isChecked():
            self.border_left_spinbox.setValue(self.border_right)
        self.update_preview()
    
    def showEvent(self, event):
        """Handle show event to update preview after widget is sized."""
        super().showEvent(event)
        # Force preview to max size (smaller size)
        if hasattr(self, 'preview_label'):
            self.preview_label.setFixedSize(450, 337)
        # Update preview after dialog is shown and widgets are sized
        if hasattr(self, 'preview_label'):
            self.update_preview()
    
    def validate_and_accept(self):
        """Validate inputs before accepting."""
        if not self.project_folder:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Validation Error", "Please select a project folder before creating the project."
            )
            return
        self.accept()

    def choose_folder(self):
        """Choose project folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder for New Project", "", QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.project_folder = Path(folder)
            self.folder_label.setText(str(self.project_folder))
            # Update default name if not set
            if not self.name_input.text():
                self.name_input.setText(self.project_folder.name)
    
    def choose_raw_images_folder(self):
        """Choose raw images folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Raw Images Folder", "", QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.raw_images_folder = Path(folder)
            self.raw_images_folder_label.setText(str(self.raw_images_folder))
    
    def get_raw_images_folder(self) -> Optional[Path]:
        """Get selected raw images folder."""
        return self.raw_images_folder

    def update_preview(self):
        """Update the tile preview display with full grid."""
        tile_w = self.tile_w_spinbox.value()
        tile_h = self.tile_h_spinbox.value()
        stride_x = self.stride_x_spinbox.value() or tile_w
        stride_y = self.stride_y_spinbox.value() or tile_h
        
        # Determine actual image dimensions to use for grid calculation
        # If we have a loaded image, use its actual size; otherwise use resolution
        if self.loaded_image_pixmap and self.loaded_image_size[0] > 0:
            # Use actual loaded image dimensions for grid calculation
            actual_img_w, actual_img_h = self.loaded_image_size
        else:
            # Get selected resolution when no image is loaded
            resolution_data = self.resolution_combo.currentData()
            if resolution_data and resolution_data[0] > 0:
                actual_img_w, actual_img_h = resolution_data
            elif self.custom_x_input.isEnabled():
                # Use custom values
                actual_img_w = self.custom_x_input.value()
                actual_img_h = self.custom_y_input.value()
            else:
                # Default to 1920x1080 if invalid
                actual_img_w, actual_img_h = 1920, 1080
        
        preview_size = self.preview_label.size()
        if preview_size.width() <= 0 or preview_size.height() <= 0:
            # Use minimum size if widget not yet sized
            preview_size = self.preview_label.minimumSize()
        
        # Calculate scale to fit preview area based on actual image dimensions
        scale_x = (preview_size.width() - 20) / actual_img_w  # Leave 10px margin on each side
        scale_y = (preview_size.height() - 20) / actual_img_h
        scale = min(scale_x, scale_y)
        
        scaled_img_w = int(actual_img_w * scale)
        scaled_img_h = int(actual_img_h * scale)
        scaled_tile_w = int(tile_w * scale)
        scaled_tile_h = int(tile_h * scale)
        
        # Create pixmap
        pixmap = QPixmap(preview_size.width(), preview_size.height())
        pixmap.fill(Qt.GlobalColor.white)
        
        painter = QPainter(pixmap)
        
        img_x = (preview_size.width() - scaled_img_w) // 2
        img_y = (preview_size.height() - scaled_img_h) // 2
        
        # Draw loaded image if available, otherwise draw gray rectangle
        if self.loaded_image_pixmap and self.loaded_image_size[0] > 0:
            # Scale and draw the actual image
            scaled_pixmap = self.loaded_image_pixmap.scaled(
                scaled_img_w,
                scaled_img_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(img_x, img_y, scaled_pixmap)
        else:
            # Draw gray rectangle for image area
            painter.setPen(QPen(Qt.GlobalColor.gray, 1))
            painter.setBrush(Qt.GlobalColor.lightGray)
            painter.drawRect(img_x, img_y, scaled_img_w, scaled_img_h)
        
        # Draw grid of tiles - use actual image dimensions for grid calculation
        painter.setPen(QPen(Qt.GlobalColor.red, 1))
        painter.setBrush(Qt.GlobalColor.transparent)
        
        # Get border values
        border_top = self.border_top_spinbox.value()
        border_bottom = self.border_bottom_spinbox.value()
        border_left = self.border_left_spinbox.value()
        border_right = self.border_right_spinbox.value()
        
        # Calculate effective image area (after borders)
        effective_img_w = actual_img_w - border_left - border_right
        effective_img_h = actual_img_h - border_top - border_bottom
        
        # Calculate grid positions starting from borders
        current_x = border_left
        current_y = border_top
        
        # Generate all tile positions that fit within the effective image area
        while current_y + tile_h <= actual_img_h - border_bottom:
            current_x = border_left
            while current_x + tile_w <= actual_img_w - border_right:
                # Calculate scaled position relative to image origin
                tile_x = img_x + int(current_x * scale)
                tile_y = img_y + int(current_y * scale)
                
                # Draw tile rectangle (bounds checking handled by while conditions)
                painter.drawRect(tile_x, tile_y, scaled_tile_w, scaled_tile_h)
                
                current_x += stride_x
            
            current_y += stride_y
        
        painter.end()
        
        self.preview_label.setPixmap(pixmap)

    def get_project_name(self) -> str:
        """Get project name."""
        return self.name_input.text().strip() or None

    def get_tile_width(self) -> int:
        """Get tile width."""
        return self.tile_w_spinbox.value()

    def get_tile_height(self) -> int:
        """Get tile height."""
        return self.tile_h_spinbox.value()

    def get_stride_x(self) -> int:
        """Get stride X (0 means use tile width)."""
        return self.stride_x_spinbox.value()

    def get_stride_y(self) -> int:
        """Get stride Y (0 means use tile height)."""
        return self.stride_y_spinbox.value()

    def get_project_folder(self) -> Optional[Path]:
        """Get selected project folder."""
        return self.project_folder

