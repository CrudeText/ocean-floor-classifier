"""Qt utility functions."""

from PIL import Image
from PySide6.QtGui import QImage, QPixmap


def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """
    Convert a PIL Image to QPixmap.

    Args:
        pil_img: PIL Image in RGB mode

    Returns:
        QPixmap suitable for display in Qt widgets
    """
    # Ensure RGB mode
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Get image data
    width, height = pil_img.size
    data = pil_img.tobytes("raw", "RGB")

    # Create QImage from data
    qimage = QImage(data, width, height, QImage.Format.Format_RGB888)

    # Convert to QPixmap
    return QPixmap.fromImage(qimage)
