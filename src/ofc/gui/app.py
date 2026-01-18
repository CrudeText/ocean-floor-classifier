"""Main GUI application."""

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main():
    """Entry point for GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Ocean Floor Classifier")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()