"""Labeling tab implementation."""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ofc.core import (
    GridSpec,
    LabelsStore,
    OceanProject,
    TileCache,
    TileRef,
    ensure_image_tiles,
    enumerate_tiles_for_image,
    get_tile_image,
)

from ..utils_qt import pil_to_qpixmap


class LabelTab(QWidget):
    """Label tab for tile labeling."""

    def __init__(self):
        """Initialize label tab."""
        super().__init__()
        self.project: Optional[OceanProject] = None
        self.grid: Optional[GridSpec] = None
        self.labels: Optional[LabelsStore] = None
        self.cache: Optional[TileCache] = None

        # Current state
        self.current_image: Optional[str] = None
        self.tiles: list[TileRef] = []
        self.current_tile_idx: int = 0
        self.classes: list[str] = []
        self.preview_folder: Optional[Path] = None  # Folder for preview mode
        self.full_image_pixmap: Optional[QPixmap] = None  # Full image for preview
        self.full_image_size: tuple[int, int] = (0, 0)  # Original image size
        # RGB filter values (0-255, where 255 = no reduction, 0 = full reduction)
        self.rgb_filter_r: int = 255
        self.rgb_filter_g: int = 255
        self.rgb_filter_b: int = 255

        self.init_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def init_ui(self):
        """Initialize UI components."""
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel: Images list
        left_panel = self.create_images_panel()
        splitter.addWidget(left_panel)

        # Center panel: Tile display
        center_panel = self.create_tile_panel()
        splitter.addWidget(center_panel)

        # Right panel: Classes and counts
        right_panel = self.create_classes_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([200, 600, 300])

    def create_images_panel(self) -> QWidget:
        """Create left panel with images list."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Title and Choose Folder button
        title_layout = QHBoxLayout()
        title = QLabel("Images")
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        choose_folder_btn = QPushButton("Choose Folder")
        choose_folder_btn.clicked.connect(self.choose_preview_folder)
        title_layout.addWidget(choose_folder_btn)
        layout.addLayout(title_layout)

        # Refresh button
        refresh_btn = QPushButton("Refresh Images")
        refresh_btn.clicked.connect(self.refresh_images)
        layout.addWidget(refresh_btn)

        # Images list
        self.images_list = QListWidget()
        self.images_list.itemSelectionChanged.connect(self.on_image_selected)
        layout.addWidget(self.images_list)

        return panel

    def create_tile_panel(self) -> QWidget:
        """Create center panel with tile display."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Image info
        self.image_info_label = QLabel("No image selected")
        layout.addWidget(self.image_info_label)

        # Tile info
        self.tile_info_label = QLabel("No tile")
        layout.addWidget(self.tile_info_label)

        # Current label
        self.current_label_label = QLabel("Label: <unlabeled>")
        layout.addWidget(self.current_label_label)

        # Use a vertical splitter to make tile and full image preview sizes relative
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section: Tile display with info panel
        tile_section = QWidget()
        tile_section_layout = QHBoxLayout()
        tile_section.setLayout(tile_section_layout)
        
        # Left: Tile display
        tile_display_widget = QWidget()
        tile_display_layout = QVBoxLayout()
        tile_display_widget.setLayout(tile_display_layout)
        
        self.tile_display = QLabel()
        self.tile_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tile_display.setMinimumSize(256, 256)
        self.tile_display.setMaximumSize(512, 512)  # Limit maximum size
        self.tile_display.setText("No tile loaded")
        tile_display_layout.addWidget(self.tile_display)
        
        # Navigation info
        nav_label = QLabel("Navigation: ←/A (prev), →/D (next), ↑/↓ (same column, change row)")
        tile_display_layout.addWidget(nav_label)
        tile_display_layout.addStretch()
        
        tile_section_layout.addWidget(tile_display_widget)
        
        # Right: Information panel
        info_panel = QWidget()
        info_panel.setMaximumWidth(250)
        info_panel.setStyleSheet("border: 1px solid gray; padding: 5px;")
        info_layout = QVBoxLayout()
        info_panel.setLayout(info_layout)
        
        info_title = QLabel("Tile Information")
        info_title.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(info_title)
        
        # File size
        self.info_file_size_label = QLabel("File Size: -")
        info_layout.addWidget(self.info_file_size_label)
        
        # Current label
        self.info_label_label = QLabel("Label: NULL")
        info_layout.addWidget(self.info_label_label)
        
        # Tile coordinates
        self.info_tile_coords_label = QLabel("Tile (i, j): -")
        info_layout.addWidget(self.info_tile_coords_label)
        
        # Tile dimensions
        self.info_tile_dims_label = QLabel("Tile Size: -")
        info_layout.addWidget(self.info_tile_dims_label)
        
        # Image dimensions
        self.info_image_dims_label = QLabel("Image Size: -")
        info_layout.addWidget(self.info_image_dims_label)
        
        # Total tiles
        self.info_total_tiles_label = QLabel("Total Tiles: -")
        info_layout.addWidget(self.info_total_tiles_label)
        
        # Tile index
        self.info_tile_index_label = QLabel("Tile Index: -")
        info_layout.addWidget(self.info_tile_index_label)
        
        info_layout.addStretch()
        tile_section_layout.addWidget(info_panel)
        
        main_splitter.addWidget(tile_section)

        # Bottom section: Full image preview with RGB sliders
        preview_section = QWidget()
        preview_section_layout = QVBoxLayout()
        preview_section.setLayout(preview_section_layout)
        
        full_image_label = QLabel("Full Image Preview")
        preview_section_layout.addWidget(full_image_label)
        
        # Horizontal layout for image and RGB sliders
        preview_layout = QHBoxLayout()
        
        # Full image display
        self.full_image_display = QLabel()
        self.full_image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.full_image_display.setMinimumHeight(200)
        self.full_image_display.setMaximumHeight(400)  # Reasonable maximum
        self.full_image_display.setText("No image loaded")
        self.full_image_display.setStyleSheet("border: 1px solid gray;")
        self.full_image_display.setScaledContents(False)  # Don't auto-scale contents
        # Override resizeEvent to update preview when widget is resized
        self.full_image_display.resizeEvent = self.on_full_image_display_resize
        preview_layout.addWidget(self.full_image_display)
        
        # RGB sliders panel (horizontal layout, compact)
        rgb_panel = QWidget()
        rgb_panel_layout = QVBoxLayout()
        rgb_panel.setLayout(rgb_panel_layout)
        rgb_panel.setMaximumWidth(120)  # Make panel more compact
        
        rgb_title = QLabel("RGB Filter")
        rgb_panel_layout.addWidget(rgb_title)
        
        # Horizontal layout for sliders (compact spacing)
        sliders_layout = QHBoxLayout()
        sliders_layout.setSpacing(2)  # Minimal spacing between sliders
        
        # Red slider
        red_widget = QWidget()
        red_widget.setMaximumWidth(35)  # Compact width
        red_layout = QVBoxLayout()
        red_widget.setLayout(red_layout)
        red_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        red_label = QLabel("R")
        red_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        red_layout.addWidget(red_label)
        self.rgb_slider_r = QSlider(Qt.Orientation.Vertical)
        self.rgb_slider_r.setMinimum(0)
        self.rgb_slider_r.setMaximum(255)
        self.rgb_slider_r.setValue(255)
        self.rgb_slider_r.valueChanged.connect(self.on_rgb_filter_changed)
        red_layout.addWidget(self.rgb_slider_r)
        red_layout.addStretch()
        sliders_layout.addWidget(red_widget)
        
        # Green slider
        green_widget = QWidget()
        green_widget.setMaximumWidth(35)  # Compact width
        green_layout = QVBoxLayout()
        green_widget.setLayout(green_layout)
        green_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        green_label = QLabel("G")
        green_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        green_layout.addWidget(green_label)
        self.rgb_slider_g = QSlider(Qt.Orientation.Vertical)
        self.rgb_slider_g.setMinimum(0)
        self.rgb_slider_g.setMaximum(255)
        self.rgb_slider_g.setValue(255)
        self.rgb_slider_g.valueChanged.connect(self.on_rgb_filter_changed)
        green_layout.addWidget(self.rgb_slider_g)
        green_layout.addStretch()
        sliders_layout.addWidget(green_widget)
        
        # Blue slider
        blue_widget = QWidget()
        blue_widget.setMaximumWidth(35)  # Compact width
        blue_layout = QVBoxLayout()
        blue_widget.setLayout(blue_layout)
        blue_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        blue_label = QLabel("B")
        blue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        blue_layout.addWidget(blue_label)
        self.rgb_slider_b = QSlider(Qt.Orientation.Vertical)
        self.rgb_slider_b.setMinimum(0)
        self.rgb_slider_b.setMaximum(255)
        self.rgb_slider_b.setValue(255)
        self.rgb_slider_b.valueChanged.connect(self.on_rgb_filter_changed)
        blue_layout.addWidget(self.rgb_slider_b)
        blue_layout.addStretch()
        sliders_layout.addWidget(blue_widget)
        
        rgb_panel_layout.addLayout(sliders_layout)
        rgb_panel_layout.addStretch()
        preview_layout.addWidget(rgb_panel)
        
        preview_section_layout.addLayout(preview_layout)
        preview_section_layout.addStretch()
        
        main_splitter.addWidget(preview_section)
        
        # Set splitter proportions (tile gets more space, preview gets less)
        main_splitter.setSizes([400, 300])
        
        layout.addWidget(main_splitter)

        return panel

    def create_classes_panel(self) -> QWidget:
        """Create right panel with classes management."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Title
        title = QLabel("Classes")
        layout.addWidget(title)

        # Add class
        add_layout = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Class name")
        self.class_input.returnPressed.connect(self.add_class)
        add_layout.addWidget(self.class_input)

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_class)
        add_layout.addWidget(add_btn)
        layout.addLayout(add_layout)

        # Classes list
        self.classes_list = QListWidget()
        layout.addWidget(self.classes_list)

        # Remove button
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_class)
        layout.addWidget(remove_btn)

        # Counts display
        layout.addWidget(QLabel("Label Counts:"))
        self.counts_label = QLabel("No labels")
        layout.addWidget(self.counts_label)

        layout.addStretch()

        return panel

    def set_project(self, project: OceanProject, grid: GridSpec, labels: LabelsStore):
        """Set the current project and update UI."""
        self.project = project
        self.grid = grid
        self.labels = labels
        self.cache = TileCache(max_items=8)

        # Set default preview folder for testing
        default_preview_folder = Path(
            r"C:\Users\willi\OneDrive\Bureau\Active Projects\OceanFloorClassifier\Usable Data\JPG"
        )
        if default_preview_folder.exists():
            self.preview_folder = default_preview_folder

        # Load classes from config
        self.load_classes()

        # Refresh images list
        self.refresh_images()

    def load_classes(self):
        """Load classes from configs/classes.json."""
        if not self.project:
            return

        classes_path = self.project.paths.configs_dir / "classes.json"
        if classes_path.exists():
            import json

            try:
                self.classes = json.loads(classes_path.read_text())
                if not isinstance(self.classes, list):
                    self.classes = []
            except Exception:
                self.classes = []
        else:
            self.classes = []

        self.update_classes_list()

    def save_classes(self):
        """Save classes to configs/classes.json."""
        if not self.project:
            return

        classes_path = self.project.paths.configs_dir / "classes.json"
        import json

        classes_path.write_text(json.dumps(self.classes, indent=2) + "\n")

    def choose_preview_folder(self):
        """Choose a folder to preview images from."""
        from PySide6.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(
            self, "Choose Folder to Preview Images", "", QFileDialog.Option.ShowDirsOnly
        )

        if folder:
            self.preview_folder = Path(folder)
            self.refresh_images()

    def refresh_images(self):
        """Refresh the images list."""
        self.images_list.clear()

        # If preview folder is set, use it; otherwise use project images
        if self.preview_folder and self.preview_folder.exists():
            try:
                from ofc.core import list_images

                image_paths = list_images(self.preview_folder)
                for img_path in image_paths:
                    # Store full path as item data, display filename
                    item = QListWidgetItem(img_path.name)
                    item.setData(Qt.ItemDataRole.UserRole, str(img_path))
                    self.images_list.addItem(item)
            except Exception as e:
                self.images_list.addItem(f"Error: {str(e)}")
        elif self.project:
            try:
                images = self.project.list_raw_images()
                for img_path in images:
                    item = QListWidgetItem(img_path)
                    self.images_list.addItem(item)
            except Exception as e:
                self.images_list.addItem(f"Error: {str(e)}")
        else:
            self.images_list.addItem("No project loaded")
        
        # Auto-select first image if available
        if self.images_list.count() > 0:
            first_item = self.images_list.item(0)
            if first_item and not first_item.text().startswith("Error:") and first_item.text() != "No project loaded":
                self.images_list.setCurrentItem(first_item)
                # Trigger selection handler
                self.on_image_selected()

    def on_image_selected(self):
        """Handle image selection."""
        selected = self.images_list.currentItem()
        if not selected:
            return

        # Check if this is a preview image (has full path in UserRole)
        full_path = selected.data(Qt.ItemDataRole.UserRole)
        is_preview = full_path is not None and full_path

        if is_preview:
            # Preview mode: use full path directly
            if not self.grid:
                return

            image_path = Path(full_path)
            if not image_path.exists():
                self.tile_display.setText(f"Image not found: {image_path}")
                return

            self.current_image = str(image_path)  # Store full path for preview
            self.current_tile_idx = 0

            # For preview images, we can't use project methods, so we need to handle differently
            # We'll create a temporary TileRef list manually
            try:
                from ofc.core import get_image_size

                width, height = get_image_size(image_path)
                self.tiles = []
                for i, j, x, y, w, h in self.grid.iter_tiles_for_image(width, height):
                    from ofc.core import TileRef

                    self.tiles.append(
                        TileRef(
                            image_rel_path=str(image_path),  # Full path for preview
                            tile_i=i,
                            tile_j=j,
                            x=x,
                            y=y,
                            w=w,
                            h=h,
                        )
                    )
                # Load full image for preview
                self.load_full_image_preview()
                self.update_tile_display()
            except Exception as e:
                self.tile_display.setText(f"Error loading tiles: {str(e)}")
        else:
            # Project image mode: use relative path
            if not self.project or not self.grid:
                return

            image_rel_path = selected.text()
            if not image_rel_path or image_rel_path.startswith("Error:"):
                return

            self.current_image = image_rel_path
            self.current_tile_idx = 0

            # Ensure labels exist for all tiles
            try:
                from ofc.core import ensure_image_tiles

                ensure_image_tiles(self.project, image_rel_path, self.grid)
                # Reload labels after ensuring tiles
                self.labels = self.project.get_labels()
            except Exception as e:
                print(f"Warning: Failed to ensure tiles: {e}")

            # Enumerate tiles
            try:
                self.tiles = enumerate_tiles_for_image(
                    self.project, image_rel_path, self.grid
                )
                # Load full image for preview
                self.load_full_image_preview()
                self.update_tile_display()
            except Exception as e:
                self.tile_display.setText(f"Error loading tiles: {str(e)}")

    def update_tile_display(self):
        """Update the tile display with current tile."""
        if (
            not self.project
            or not self.tiles
            or self.current_tile_idx < 0
            or self.current_tile_idx >= len(self.tiles)
        ):
            self.tile_display.setText("No tile to display")
            return

        tile = self.tiles[self.current_tile_idx]

        # Update info labels
        if self.current_image:
            self.image_info_label.setText(f"Image: {self.current_image}")
        self.tile_info_label.setText(f"Tile ({tile.tile_i}, {tile.tile_j})")

        # Get current label (only for project images, not preview)
        is_preview = Path(tile.image_rel_path).is_absolute()
        if is_preview:
            self.current_label_label.setText("Label: <preview mode - cannot label>")
            label_text = "NULL"
        elif self.labels:
            label = self.labels.get(self.current_image, tile.tile_i, tile.tile_j)
            if label:
                self.current_label_label.setText(f"Label: {label}")
                label_text = label
            else:
                self.current_label_label.setText("Label: <unlabeled>")
                label_text = "NULL"
        else:
            self.current_label_label.setText("Label: <unlabeled>")
            label_text = "NULL"
        
        # Update information panel
        self.update_info_panel(tile, label_text, is_preview)

        # Load and display tile image
        try:
            # Check if this is a preview image (full path) or project image
            is_preview = Path(tile.image_rel_path).is_absolute()
            
            if is_preview:
                # Preview mode: load image directly
                from ofc.core import read_image_pil
                from PIL import Image

                base_img = read_image_pil(Path(tile.image_rel_path))
                if self.cache:
                    # Cache the full path image
                    self.cache._cache[Path(tile.image_rel_path)] = base_img
                
                # Crop the tile
                x, y, w, h = tile.x, tile.y, tile.w, tile.h
                img_width, img_height = base_img.size
                
                # Handle padding if needed
                if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                    # Create padded image
                    tile_img = Image.new("RGB", (w, h), (0, 0, 0))
                    crop_x = max(0, x)
                    crop_y = max(0, y)
                    crop_x_end = min(img_width, x + w)
                    crop_y_end = min(img_height, y + h)
                    paste_x = crop_x - x
                    paste_y = crop_y - y
                    if crop_x < crop_x_end and crop_y < crop_y_end:
                        cropped = base_img.crop((crop_x, crop_y, crop_x_end, crop_y_end))
                        tile_img.paste(cropped, (paste_x, paste_y))
                else:
                    tile_img = base_img.crop((x, y, x + w, y + h))
            else:
                # Project mode: use existing method
                tile_img = get_tile_image(
                    self.project, tile, cache=self.cache, pad_value=(0, 0, 0)
                )
            
            pixmap = pil_to_qpixmap(tile_img)
            # Apply RGB filter
            filtered_pixmap = self.apply_rgb_filter(pixmap)
            # Scale to fit display while maintaining aspect ratio
            scaled_pixmap = filtered_pixmap.scaled(
                self.tile_display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.tile_display.setPixmap(scaled_pixmap)
            
            # Update full image preview with tile highlight
            self.update_full_image_preview()
        except Exception as e:
            self.tile_display.setText(f"Error loading tile: {str(e)}")

    def update_info_panel(self, tile: TileRef, label_text: str, is_preview: bool):
        """Update the information panel with tile and project details."""
        # File size
        try:
            if is_preview:
                image_path = Path(tile.image_rel_path)
            else:
                if not self.project:
                    return
                image_path = self.project.get_raw_image_path(self.current_image)
            
            if image_path.exists():
                file_size = image_path.stat().st_size
                # Format file size
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.2f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.2f} MB"
                self.info_file_size_label.setText(f"File Size: {size_str}")
            else:
                self.info_file_size_label.setText("File Size: -")
        except Exception:
            self.info_file_size_label.setText("File Size: -")
        
        # Current label
        self.info_label_label.setText(f"Label: {label_text}")
        
        # Tile coordinates
        self.info_tile_coords_label.setText(f"Tile (i, j): ({tile.tile_i}, {tile.tile_j})")
        
        # Tile dimensions
        self.info_tile_dims_label.setText(f"Tile Size: {tile.w} × {tile.h} px")
        
        # Image dimensions
        try:
            if is_preview:
                from ofc.core import get_image_size
                img_width, img_height = get_image_size(Path(tile.image_rel_path))
            else:
                if not self.project:
                    return
                image_path = self.project.get_raw_image_path(self.current_image)
                from ofc.core import get_image_size
                img_width, img_height = get_image_size(image_path)
            self.info_image_dims_label.setText(f"Image Size: {img_width} × {img_height} px")
        except Exception:
            self.info_image_dims_label.setText("Image Size: -")
        
        # Total tiles
        if self.tiles:
            self.info_total_tiles_label.setText(f"Total Tiles: {len(self.tiles)}")
        else:
            self.info_total_tiles_label.setText("Total Tiles: -")
        
        # Tile index (1-based for user display)
        self.info_tile_index_label.setText(f"Tile Index: {self.current_tile_idx + 1} / {len(self.tiles) if self.tiles else 0}")

    def load_full_image_preview(self):
        """Load the full image for preview display."""
        if not self.current_image or not self.tiles:
            self.full_image_pixmap = None
            self.full_image_display.setText("No image loaded")
            return

        try:
            from ofc.core import read_image_pil

            # Determine image path
            is_preview = Path(self.current_image).is_absolute()
            if is_preview:
                image_path = Path(self.current_image)
            else:
                if not self.project:
                    return
                image_path = self.project.get_raw_image_path(self.current_image)

            if not image_path.exists():
                self.full_image_display.setText("Image not found")
                return

            # Load full image
            full_img = read_image_pil(image_path)
            self.full_image_size = full_img.size  # Store original size
            
            # Convert to QPixmap
            self.full_image_pixmap = pil_to_qpixmap(full_img)
            
            # Update display
            self.update_full_image_preview()
        except Exception as e:
            self.full_image_display.setText(f"Error loading image: {str(e)}")
            self.full_image_pixmap = None

    def update_full_image_preview(self):
        """Update the full image preview with highlighted current tile."""
        if not self.full_image_pixmap or not self.tiles or self.current_tile_idx < 0 or self.current_tile_idx >= len(self.tiles):
            return

        tile = self.tiles[self.current_tile_idx]
        
        # Get actual widget size (not maximum, but current available size)
        widget_width = self.full_image_display.width()
        widget_height = self.full_image_display.height()
        
        # If widget hasn't been sized yet, skip update (will be called again on resize)
        if widget_width <= 0 or widget_height <= 0:
            return

        # Calculate scaling factor to fit the image within widget bounds
        img_width, img_height = self.full_image_size
        scale_x = widget_width / img_width
        scale_y = widget_height / img_height
        scale = min(scale_x, scale_y)  # Maintain aspect ratio, fit within bounds (allow upscale if needed)
        
        # Calculate final scaled dimensions (always scale to fit, can be smaller than widget)
        scaled_width = int(img_width * scale)
        scaled_height = int(img_height * scale)
        
        # Scale the pixmap to fit within widget bounds
        scaled_pixmap = self.full_image_pixmap.scaled(
            scaled_width,
            scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        
        # Create a copy to draw on
        preview_pixmap = scaled_pixmap.copy()
        
        # Draw highlighted rectangle for current tile
        painter = QPainter(preview_pixmap)
        pen = QPen(Qt.GlobalColor.red, max(1, int(3 * scale)))  # Scale pen width with image
        painter.setPen(pen)
        
        # Calculate tile rectangle in scaled coordinates
        tile_x = int(tile.x * scale)
        tile_y = int(tile.y * scale)
        tile_w = int(tile.w * scale)
        tile_h = int(tile.h * scale)
        
        # Ensure tile coordinates are within pixmap bounds
        tile_x = max(0, min(tile_x, scaled_width - 1))
        tile_y = max(0, min(tile_y, scaled_height - 1))
        tile_w = min(tile_w, scaled_width - tile_x)
        tile_h = min(tile_h, scaled_height - tile_y)
        
        # Draw rectangle
        painter.drawRect(tile_x, tile_y, tile_w, tile_h)
        painter.end()
        
        # Apply RGB filter to the preview
        filtered_preview = self.apply_rgb_filter(preview_pixmap)
        
        # Display the preview (this won't resize the widget because setScaledContents is False)
        self.full_image_display.setPixmap(filtered_preview)

    def on_rgb_filter_changed(self):
        """Handle RGB filter slider changes."""
        # Update filter values from sliders
        self.rgb_filter_r = self.rgb_slider_r.value()
        self.rgb_filter_g = self.rgb_slider_g.value()
        self.rgb_filter_b = self.rgb_slider_b.value()
        
        # Refresh both displays
        if self.tiles and self.current_tile_idx >= 0:
            self.update_tile_display()
            self.update_full_image_preview()

    def apply_rgb_filter(self, pixmap: QPixmap) -> QPixmap:
        """Apply RGB filter to a pixmap by reducing RGB values."""
        if self.rgb_filter_r == 255 and self.rgb_filter_g == 255 and self.rgb_filter_b == 255:
            # No filtering needed
            return pixmap
        
        # Convert pixmap to image for pixel manipulation
        image = pixmap.toImage()
        
        # Create filtered image
        filtered_image = QImage(image.size(), QImage.Format.Format_RGB32)
        
        for y in range(image.height()):
            for x in range(image.width()):
                pixel = image.pixel(x, y)
                # Extract RGB values
                r = (pixel >> 16) & 0xFF
                g = (pixel >> 8) & 0xFF
                b = pixel & 0xFF
                
                # Apply filter (reduce values proportionally)
                r = int(r * self.rgb_filter_r / 255)
                g = int(g * self.rgb_filter_g / 255)
                b = int(b * self.rgb_filter_b / 255)
                
                # Clamp values
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                # Set filtered pixel
                filtered_pixel = (r << 16) | (g << 8) | b | 0xFF000000
                filtered_image.setPixel(x, y, filtered_pixel)
        
        return QPixmap.fromImage(filtered_image)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        key = event.key()

        # Number keys 1-9: assign class
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            class_idx = key - Qt.Key.Key_1
            if class_idx < len(self.classes):
                self.assign_label(self.classes[class_idx])
            return

        # 0: unlabel
        if key == Qt.Key.Key_0:
            self.assign_label("")
            return

        # Arrow keys and WASD for navigation
        if key == Qt.Key.Key_Right or key == Qt.Key.Key_D:
            self.next_tile()
            return
        if key == Qt.Key.Key_Left or key == Qt.Key.Key_A:
            self.prev_tile()
            return
        if key == Qt.Key.Key_Up:
            self.move_vertical(-1)  # Move up one row (same column)
            return
        if key == Qt.Key.Key_Down:
            self.move_vertical(1)  # Move down one row (same column)
            return

        super().keyPressEvent(event)

    def next_tile(self):
        """Move to next tile."""
        if self.tiles and self.current_tile_idx < len(self.tiles) - 1:
            self.current_tile_idx += 1
            self.update_tile_display()

    def prev_tile(self):
        """Move to previous tile."""
        if self.current_tile_idx > 0:
            self.current_tile_idx -= 1
            self.update_tile_display()

    def move_vertical(self, direction: int):
        """Move vertically in the grid (same column, different row).
        
        Args:
            direction: -1 for up, 1 for down
        """
        if not self.tiles or self.current_tile_idx < 0 or self.current_tile_idx >= len(self.tiles):
            return
        
        current_tile = self.tiles[self.current_tile_idx]
        current_i = current_tile.tile_i
        current_j = current_tile.tile_j
        
        # Find all tiles in the same column (same i)
        same_column_tiles = [
            (idx, tile) for idx, tile in enumerate(self.tiles)
            if tile.tile_i == current_i
        ]
        
        if not same_column_tiles:
            return
        
        # Sort by j (row) to find next/previous
        same_column_tiles.sort(key=lambda x: x[1].tile_j)
        
        # Find current position in this column
        current_pos = None
        for pos, (idx, tile) in enumerate(same_column_tiles):
            if idx == self.current_tile_idx:
                current_pos = pos
                break
        
        if current_pos is None:
            return
        
        # Move to next/previous row
        new_pos = current_pos + direction
        if 0 <= new_pos < len(same_column_tiles):
            new_idx = same_column_tiles[new_pos][0]
            self.current_tile_idx = new_idx
            self.update_tile_display()

    def assign_label(self, label: str):
        """Assign label to current tile and save."""
        # Only allow labeling for project images, not preview images
        if (
            not self.project
            or not self.labels
            or not self.current_image
            or not self.tiles
            or self.current_tile_idx < 0
            or self.current_tile_idx >= len(self.tiles)
        ):
            return

        # Check if this is a preview image (can't label preview images)
        tile = self.tiles[self.current_tile_idx]
        is_preview = Path(tile.image_rel_path).is_absolute()
        if is_preview:
            # Preview mode: don't allow labeling
            return

        self.labels.set(self.current_image, tile.tile_i, tile.tile_j, label)

        # Save immediately
        self.labels.save(self.project.paths.data_labels)

        # Update display
        self.update_tile_display()
        self.update_counts()

    def add_class(self):
        """Add a new class."""
        class_name = self.class_input.text().strip()
        if not class_name:
            return

        if class_name not in self.classes:
            self.classes.append(class_name)
            self.save_classes()
            self.update_classes_list()

        self.class_input.clear()

    def remove_class(self):
        """Remove selected class."""
        selected = self.classes_list.currentItem()
        if not selected:
            return

        class_name = selected.text()
        if class_name in self.classes:
            self.classes.remove(class_name)
            self.save_classes()
            self.update_classes_list()

    def update_classes_list(self):
        """Update the classes list widget."""
        self.classes_list.clear()
        for i, class_name in enumerate(self.classes):
            item = QListWidgetItem(f"{i+1}. {class_name}")
            self.classes_list.addItem(item)

    def update_counts(self):
        """Update the label counts display."""
        if not self.labels:
            self.counts_label.setText("No labels")
            return

        counts = self.labels.counts()
        if not counts:
            self.counts_label.setText("No labels")
            return

        lines = []
        for label, count in sorted(counts.items()):
            label_display = label if label else "<unlabeled>"
            lines.append(f"{label_display}: {count}")

        self.counts_label.setText("\n".join(lines))

    def resizeEvent(self, event):
        """Handle resize to update tile display scaling."""
        super().resizeEvent(event)
        if self.tiles and self.current_tile_idx >= 0:
            self.update_tile_display()
            # Don't update full image preview here - it will be updated by its own resize event

    def on_full_image_display_resize(self, event):
        """Handle resize of full image display widget."""
        # Call the original resizeEvent for QLabel
        QLabel.resizeEvent(self.full_image_display, event)
        # Update the preview to match new size
        if self.full_image_pixmap and self.tiles and self.current_tile_idx >= 0:
            self.update_full_image_preview()