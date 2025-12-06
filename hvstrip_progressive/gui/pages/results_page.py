"""
Results Page - View Analysis Results
Provides interface for browsing and viewing workflow results
"""

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QLabel
)
from PySide6.QtGui import QPixmap
from qfluentwidgets import (
    ScrollArea,
    TitleLabel,
    BodyLabel,
    PushButton,
    ListWidget,
    CardWidget,
    FluentIcon,
    InfoBar,
    InfoBarPosition
)


class ResultsPage(ScrollArea):
    """Results page for viewing analysis outputs"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_directory = None
        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        # Create scroll widget
        self.scroll_widget = QWidget()
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)
        self.setObjectName("resultsPage")

        # Main layout
        layout = QVBoxLayout(self.scroll_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Page title
        title = TitleLabel("Results Viewer")
        layout.addWidget(title)

        # Description
        desc = BodyLabel(
            "Browse and view workflow results. Select an output directory to explore generated files."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_label = BodyLabel("No directory selected")
        self.browse_button = PushButton("Browse Output Directory", self, FluentIcon.FOLDER)
        self.browse_button.clicked.connect(self.browse_directory)

        dir_layout.addWidget(self.dir_label)
        dir_layout.addStretch()
        dir_layout.addWidget(self.browse_button)
        layout.addLayout(dir_layout)

        # Content area - split between file list and preview
        content_layout = QHBoxLayout()

        # File list panel
        file_panel = CardWidget()
        file_layout = QVBoxLayout(file_panel)
        file_layout.setContentsMargins(15, 15, 15, 15)

        file_title = BodyLabel("Generated Files")
        file_title.setStyleSheet("font-weight: bold;")
        file_layout.addWidget(file_title)

        self.file_list = ListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        file_layout.addWidget(self.file_list)

        # Action buttons
        action_layout = QHBoxLayout()
        self.open_button = PushButton("Open File", self, FluentIcon.DOCUMENT)
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_selected_file)

        self.open_folder_button = PushButton("Open Folder", self, FluentIcon.FOLDER)
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self.open_in_explorer)

        action_layout.addWidget(self.open_button)
        action_layout.addWidget(self.open_folder_button)
        action_layout.addStretch()

        file_layout.addLayout(action_layout)

        content_layout.addWidget(file_panel, 1)

        # Preview panel
        preview_panel = CardWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(15, 15, 15, 15)

        preview_title = BodyLabel("Preview")
        preview_title.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_title)

        # Image preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.preview_label.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
        self.preview_label.setText("Select an image file to preview")

        # Scroll area for image
        preview_scroll = ScrollArea()
        preview_scroll.setWidget(self.preview_label)
        preview_scroll.setWidgetResizable(True)
        preview_layout.addWidget(preview_scroll)

        content_layout.addWidget(preview_panel, 2)

        layout.addLayout(content_layout)

    def browse_directory(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )

        if directory:
            self.load_directory(directory)

    def load_directory(self, directory):
        """Load files from directory"""
        self.current_directory = Path(directory)
        self.dir_label.setText(f"Directory: {directory}")

        # Clear file list
        self.file_list.clear()

        # Find all relevant files
        file_patterns = [
            '**/*.png',
            '**/*.pdf',
            '**/*.csv',
            '**/*.txt',
            '**/*.json'
        ]

        files = []
        for pattern in file_patterns:
            files.extend(self.current_directory.glob(pattern))

        # Sort files by name
        files = sorted(files, key=lambda x: x.name)

        # Add to list with relative paths
        for file in files:
            rel_path = file.relative_to(self.current_directory)
            self.file_list.addItem(str(rel_path))

        # Enable buttons if files found
        if files:
            self.open_folder_button.setEnabled(True)
            InfoBar.success(
                "Success",
                f"Loaded {len(files)} files from directory",
                duration=2000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
        else:
            InfoBar.warning(
                "Warning",
                "No result files found in selected directory",
                duration=2000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )

    def on_file_selected(self, item):
        """Handle file selection"""
        if not self.current_directory:
            return

        file_path = self.current_directory / item.text()
        self.open_button.setEnabled(True)

        # Try to preview if it's an image
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            self.preview_image(file_path)
        else:
            self.preview_label.setText(f"Preview not available for {file_path.suffix} files")

    def preview_image(self, file_path):
        """Preview an image file"""
        pixmap = QPixmap(str(file_path))

        if not pixmap.isNull():
            # Scale to fit while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                800, 800,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
            self.preview_label.setMinimumSize(scaled_pixmap.size())
        else:
            self.preview_label.setText("Failed to load image")

    def open_selected_file(self):
        """Open selected file with default application"""
        current_item = self.file_list.currentItem()
        if not current_item or not self.current_directory:
            return

        file_path = self.current_directory / current_item.text()

        # Open file with default application
        import subprocess
        import platform

        try:
            if platform.system() == 'Windows':
                subprocess.Popen(['start', str(file_path)], shell=True)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['open', str(file_path)])
            else:  # Linux
                subprocess.Popen(['xdg-open', str(file_path)])

            InfoBar.success(
                "Success",
                "Opening file...",
                duration=1000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
        except Exception as e:
            InfoBar.error(
                "Error",
                f"Failed to open file: {str(e)}",
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )

    def open_in_explorer(self):
        """Open directory in file explorer"""
        if not self.current_directory:
            return

        import subprocess
        import platform

        try:
            if platform.system() == 'Windows':
                subprocess.Popen(['explorer', str(self.current_directory)])
            elif platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['open', str(self.current_directory)])
            else:  # Linux
                subprocess.Popen(['xdg-open', str(self.current_directory)])

            InfoBar.success(
                "Success",
                "Opening folder...",
                duration=1000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
        except Exception as e:
            InfoBar.error(
                "Error",
                f"Failed to open folder: {str(e)}",
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
