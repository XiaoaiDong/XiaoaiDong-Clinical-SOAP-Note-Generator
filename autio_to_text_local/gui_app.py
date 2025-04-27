# gui_app.py
import sys
# import asyncio # No longer needed by worker
from pathlib import Path
import traceback
import datetime

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QCheckBox, QTextEdit, QProgressBar,
    QSizePolicy, QSpacerItem, QMessageBox, QGroupBox, QFrame
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject, QSize
from PyQt6.QtGui import QFont, QIcon, QPixmap

# --- Import from backend.py ---
try:
    # Import necessary components from the backend module
    from backend import (
        FileTranscriberTranslator, # The main processing class
        PYANNOTE_INSTALLED,      # Status flag for Pyannote
        PYDUB_READY,             # Status flag for Pydub/FFmpeg
        format_timestamp,        # Helper function for timestamps
        OUTPUT_DIR               # Default directory for saving files
    )
    BACKEND_AVAILABLE = True # Flag indicating backend loaded successfully
except ImportError as e:
     # Handle case where backend.py or its dependencies are missing
     print(f"Error: Failed to import required components from backend.py: {e}")
     print("Please ensure backend.py is in the same directory as gui_app.py and all necessary libraries (Whisper, PyTorch, Pyannote, Pydub, OpenAI, FPDF, PyQt6) are installed.")
     BACKEND_AVAILABLE = False # Set flag to indicate backend failure
     # The application will show an error and exit later if this is False

# --- Background Processing Thread ---
class WorkerSignals(QObject):
    """Defines signals available from the background worker thread for UI updates."""
    progress = pyqtSignal(int)       # Reports processing progress (percentage)
    status = pyqtSignal(str, bool)   # Reports status messages (text, is_error_flag)
    finished = pyqtSignal(bool, object) # Signals task completion (success_flag, result_data)
    error = pyqtSignal(str)          # Reports critical errors encountered during processing

class Worker(QThread):
    """Worker thread to execute the audio processing task in the background."""
    def __init__(self, processor: FileTranscriberTranslator, file_path: str):
        """Initializes the worker with the processor instance and file path."""
        super().__init__()
        self.processor = processor
        self.file_path = file_path
        self.signals = WorkerSignals() # Instantiate signals object

        # Connect the processor's internal callbacks (if they exist) to this thread's signals
        # This allows the backend to send status/progress updates back to the GUI via the worker thread
        if self.processor:
             # Use lambda to ensure signals are emitted correctly from this QThread instance
             self.processor.status_callback = lambda msg, is_err=False: self.signals.status.emit(msg, is_err)
             self.processor.progress_callback = self.signals.progress.emit

    def run(self):
        """The main execution logic for the worker thread."""
        if not self.processor:
            self.signals.error.emit("Backend processor instance is missing."); return # Error if processor wasn't passed correctly
        try:
            # --- Execute the main processing function from the backend ---
            # This is a synchronous call within this background thread
            success = self.processor.process_audio_file(self.file_path)
            # -----------------------------------------------------------

            # Retrieve the final result (expected attribute name in the processor)
            final_result = getattr(self.processor, 'final_output_segments', None)
            # Emit the 'finished' signal with success status and results
            self.signals.finished.emit(success, final_result)

        except Exception as e:
            # Catch any unexpected exceptions during backend processing
            error_message = f"A critical error occurred during background audio processing: {e}\n{traceback.format_exc()}"
            self.signals.error.emit(error_message) # Emit the error signal
        finally:
            # --- Cleanup ---
            # Disconnect callbacks in the processor to prevent potential issues if the processor is reused
            if self.processor:
                self.processor.status_callback = None
                self.processor.progress_callback = None

# --- Main Application Window ---
class MainWindow(QMainWindow):
    """The main GUI window for the audio transcription application."""
    def __init__(self):
        """Initializes the main window, checks dependencies, and sets up the UI."""
        super().__init__()
        self.setWindowTitle("Intelligent Audio Transcription & Translation Tool v2.3 (Local Whisper + TXT Save)") # Updated title
        self.setGeometry(100, 100, 850, 700) # Initial window size and position

        # --- Set Application Icon ---
        # !! Replace with your actual icon path or remove if not needed !!
        icon_path = 'C:/Users/59321/autio_to_text/icons/app_icon.png' # Example path
        if Path(icon_path).exists():
             self.setWindowIcon(QIcon(icon_path))
        else:
             print(f"Warning: Application icon file not found at '{icon_path}'")

        # --- Initialize Member Variables ---
        self.processor = None             # Holds the backend processor instance
        self.selected_file_path = None    # Stores the path of the currently selected audio file
        self.processing_thread = None     # Holds the background worker thread instance
        self.last_successful_result = None # Stores the data from the last successful run for saving

        # --- Critical Backend Check ---
        if not BACKEND_AVAILABLE:
             # Show an error and prevent further initialization if backend failed to load
             self._show_critical_error("Backend Module Load Failed",
                                       "Could not load the backend.py module or its dependencies.\n"
                                       "Please ensure the file exists and all required libraries are installed.\n"
                                       "The application cannot start.")
             # We might want to exit immediately or disable all functionality
             # For now, returning prevents UI setup.
             return

        # --- Check Optional Dependencies (Pyannote, Pydub) ---
        if not PYANNOTE_INSTALLED or not PYDUB_READY:
            warning_message = "Warning: Some optional features may be limited due to missing dependencies:\n\n"
            if not PYANNOTE_INSTALLED:
                warning_message += "[-] Speaker Diarization: Requires 'pyannote.audio' and 'torch'. Speaker labels will be 'UNKNOWN'.\n"
            if not PYDUB_READY:
                warning_message += "[-] Audio Conversion/Chunking: Requires 'pydub' and 'ffmpeg'. Only WAV/FLAC files might work directly.\n"
            # Show a non-blocking warning message
            QMessageBox.warning(self, "Optional Dependency Warning", warning_message)

        # --- Initialize Backend Processor and UI ---
        try:
            # Create an instance of the backend processor
            # Initial translation state is off; can be changed by checkbox
            self.processor = FileTranscriberTranslator(translate=False)

            # Check if models loaded correctly within the processor
            if not self.processor.local_whisper_model:
                 # This indicates an error during Whisper model loading inside the backend __init__
                 raise RuntimeError("Local Whisper model failed to load during backend initialization.")
            if PYANNOTE_INSTALLED and not getattr(self.processor, 'diarization_pipeline', None):
                 # Pyannote library is installed, but the pipeline model failed to load
                 print("Warning: Pyannote library detected, but the diarization pipeline model could not be loaded (check token/connection?).")

            # If processor initialization seems okay, build the UI
            self._create_ui_elements() # Create all widgets and layouts
            self._connect_signals()    # Connect button clicks etc. to functions
            self._apply_styles()       # Apply CSS styling
            self.update_status("Ready. Please select an audio file to process.", is_error=False) # Set initial status

        except Exception as e:
             # Catch errors during processor or UI initialization
             error_msg = f"Failed to initialize the application:\n{e}\n\n" \
                         f"Common causes:\n" \
                         f"- Missing dependencies (Whisper, PyTorch, etc.)\n" \
                         f"- Incorrect API Key or Hugging Face Token in backend.py\n" \
                         f"- Issues downloading models (check internet connection)\n" \
                         f"- Font file path errors in backend.py"
             self._show_critical_error("Application Initialization Failed", error_msg)
             # Attempt to show a minimal UI even on failure, but disable controls
             self._create_ui_elements()
             self._apply_styles()
             if hasattr(self, 'process_button'): self.process_button.setEnabled(False)
             if hasattr(self, 'select_button'): self.select_button.setEnabled(False)
             if hasattr(self, 'save_pdf_button'): self.save_pdf_button.setEnabled(False)
             if hasattr(self, 'save_txt_button'): self.save_txt_button.setEnabled(False)


    def _show_critical_error(self, title, message):
        """Helper function to display a critical error message dialog."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical) # Set critical icon
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec() # Show the dialog modally

    def _create_ui_elements(self):
        """Creates and arranges all the GUI widgets."""
        self.central_widget = QWidget() # Main container widget
        self.setCentralWidget(self.central_widget)
        # Use QGridLayout for flexible arrangement
        self.main_layout = QGridLayout(self.central_widget)
        self.main_layout.setSpacing(15) # Spacing between widgets
        self.main_layout.setContentsMargins(20, 20, 20, 20) # Margins around the grid

        # --- Row 1: File Selection & Options ---
        file_groupbox = QGroupBox("File & Options") # Group box for organization
        file_layout = QHBoxLayout() # Horizontal layout inside group box
        # Select File Button
        # !! Replace with your actual icon paths !!
        folder_icon_path = 'path/to/your/folder_icon.png' # Example path
        select_icon = QIcon(folder_icon_path) if Path(folder_icon_path).exists() else QIcon() # Load icon if exists
        self.select_button = QPushButton(select_icon, " Select File")
        self.select_button.setIconSize(QSize(18, 18)); self.select_button.setToolTip("Click to select an audio file (mp3, wav, flac, m4a, etc.)"); self.select_button.setFixedHeight(35)
        # File Label (shows selected filename)
        self.file_label = QLabel("No file selected yet"); self.file_label.setObjectName("file_label") # Object name for styling
        # Translation Checkbox
        self.translate_checkbox = QCheckBox("Enable Chinese Translation (uses API)"); self.translate_checkbox.setToolTip("If checked, non-Chinese text will be translated using the OpenAI API"); self.translate_checkbox.setChecked(False) # Default off
        # Add widgets to file layout
        file_layout.addWidget(self.select_button); file_layout.addWidget(self.file_label, 1); # Label stretches
        file_layout.addWidget(self.translate_checkbox); file_groupbox.setLayout(file_layout)
        self.main_layout.addWidget(file_groupbox, 1, 0, 1, 2) # Span 2 columns

        # --- Row 2: Control Buttons ---
        control_layout = QHBoxLayout(); control_layout.addStretch() # Align buttons to the right
        # Process Button
        process_icon_path = 'path/to/your/process_icon.png' # Example path
        process_icon = QIcon(process_icon_path) if Path(process_icon_path).exists() else QIcon()
        self.process_button = QPushButton(process_icon, " Start Processing"); self.process_button.setObjectName("process_button"); self.process_button.setIconSize(QSize(20, 20)); self.process_button.setFixedHeight(40); self.process_button.setEnabled(False); self.process_button.setToolTip("Start transcription and optional translation using the local Whisper model")
        # Save PDF Button
        save_pdf_icon_path = 'path/to/your/pdf_icon.png' # Example PDF icon path
        save_pdf_icon = QIcon(save_pdf_icon_path) if Path(save_pdf_icon_path).exists() else QIcon()
        self.save_pdf_button = QPushButton(save_pdf_icon, " Save PDF"); self.save_pdf_button.setObjectName("save_button"); self.save_pdf_button.setIconSize(QSize(18, 18)); self.save_pdf_button.setFixedHeight(40); self.save_pdf_button.setEnabled(False); self.save_pdf_button.setToolTip("Save the processed result as a PDF file")
        # Save TXT Button (New)
        save_txt_icon_path = 'path/to/your/txt_icon.png' # Example TXT icon path
        save_txt_icon = QIcon(save_txt_icon_path) if Path(save_txt_icon_path).exists() else QIcon()
        self.save_txt_button = QPushButton(save_txt_icon, " Save TXT"); self.save_txt_button.setObjectName("save_txt_button"); self.save_txt_button.setIconSize(QSize(18, 18)); self.save_txt_button.setFixedHeight(40); self.save_txt_button.setEnabled(False); self.save_txt_button.setToolTip("Save the processed result as a plain text (TXT) file")
        # Add buttons to control layout
        control_layout.addWidget(self.process_button)
        control_layout.addWidget(self.save_pdf_button) # Changed name here for clarity
        control_layout.addWidget(self.save_txt_button) # Added new TXT button
        # Add control layout to main grid
        control_widget = QWidget(); control_widget.setLayout(control_layout)
        self.main_layout.addWidget(control_widget, 2, 0, 1, 2) # Span 2 columns

        # --- Row 3: Status Label and Progress Bar ---
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Ready."); self.status_label.setObjectName("status_label")
        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0); self.progress_bar.setTextVisible(False); self.progress_bar.setVisible(False) # Initially hidden
        status_layout.addWidget(self.status_label, 1); status_layout.addWidget(self.progress_bar)
        status_widget = QWidget(); status_widget.setLayout(status_layout)
        self.main_layout.addWidget(status_widget, 3, 0, 1, 2) # Span 2 columns

        # --- Row 4: Result Display Area ---
        result_frame = QFrame(); result_frame.setObjectName("result_frame"); result_frame.setFrameShape(QFrame.Shape.StyledPanel) # Add frame for visual separation
        result_layout = QVBoxLayout(result_frame); result_layout.setContentsMargins(5, 5, 5, 5)
        self.result_label = QLabel("Processing Result:") # Label for the text area
        self.result_display = QTextEdit(); self.result_display.setReadOnly(True) # Read-only text area for results
        # Set monospaced font for better alignment of results
        result_font = QFont("Courier New", 10); # Preferred monospaced font
        if result_font.family() != "Courier New": result_font = QFont("Monospace", 10) # Fallback generic monospaced
        self.result_display.setFont(result_font); self.result_display.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap) # Disable line wrapping
        result_layout.addWidget(self.result_label); result_layout.addWidget(self.result_display, 1) # Text area stretches
        self.main_layout.addWidget(result_frame, 4, 0, 1, 2) # Span 2 columns

        # --- Layout Stretching ---
        # Make the result area (row 4) expand vertically
        self.main_layout.setRowStretch(4, 1);
        # Optional: Make columns stretch equally
        self.main_layout.setColumnStretch(0, 1); self.main_layout.setColumnStretch(1, 1)

    def _connect_signals(self):
         """Connects widget signals (e.g., button clicks) to their handler functions (slots)."""
         # Check if elements exist before connecting (important if init failed partially)
         if hasattr(self, 'select_button'): self.select_button.clicked.connect(self.select_file)
         if hasattr(self, 'process_button'): self.process_button.clicked.connect(self.start_processing)
         # Connect SAVE buttons
         if hasattr(self, 'save_pdf_button'): self.save_pdf_button.clicked.connect(self.save_pdf)
         if hasattr(self, 'save_txt_button'): self.save_txt_button.clicked.connect(self.save_txt) # Connect new button's signal
         # ---------------------
         if hasattr(self, 'translate_checkbox'): self.translate_checkbox.stateChanged.connect(self.update_translation_option)
         # Worker signals are connected when the worker is started in start_processing

    def _apply_styles(self):
         """Applies a QSS stylesheet for custom widget appearances."""
         # Add or modify styles for new elements like save_txt_button if needed
         style_sheet = """
            QMainWindow { background-color: #f8f9fa; } /* Light gray background */
            QGroupBox {
                font-size: 11pt;
                border: 1px solid #dee2e6; /* Light border */
                border-radius: 6px;
                margin-top: 10px; /* Space above the box */
                background-color: #ffffff; /* White background */
                padding: 10px 10px 10px 10px; /* Padding inside */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px 0 5px; /* Padding around title */
                left: 10px; /* Position from left */
                color: #495057; /* Title color */
            }
            QLabel { font-size: 10pt; color: #495057; } /* Default label style */
            QLabel#file_label { color: #6c757d; font-style: italic; padding-left: 5px; } /* Style for file label */
            QLabel#status_label { padding-left: 5px; } /* Style for status label */

            /* Default Button Style */
            QPushButton {
                font-size: 10pt; color: white;
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #007bff, stop:1 #0056b3); /* Blue gradient */
                border: 1px solid #0056b3; padding: 8px 15px; border-radius: 4px;
                min-width: 110px; /* Minimum button width */
                outline: none; /* Remove focus outline */
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0069d9, stop:1 #004085); /* Darker blue on hover */
                border: 1px solid #004085;
            }
            QPushButton:pressed { background-color: #004085; } /* Even darker when pressed */
            QPushButton:disabled {
                background-color: #e9ecef; border: 1px solid #ced4da; color: #6c757d; /* Grayed out when disabled */
            }

            /* PDF Save Button Style (Green) */
            QPushButton#save_button { /* Specific style for PDF save button */
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #28a745, stop:1 #218838); /* Green gradient */
                border: 1px solid #1e7e34;
            }
            QPushButton#save_button:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #218838, stop:1 #1e7e34); /* Darker green */
                border: 1px solid #1c7430;
            }
            QPushButton#save_button:pressed { background-color: #1e7e34; }
            /* Disabled state inherits general QPushButton disabled style */

            /* TXT Save Button Style (Cyan/Teal Example) */
            QPushButton#save_txt_button { /* Specific style for TXT save button */
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #17a2b8, stop:1 #117a8b); /* Cyan gradient */
                border: 1px solid #10707f;
            }
            QPushButton#save_txt_button:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #138496, stop:1 #0f6674); /* Darker cyan */
                border: 1px solid #0f6674;
            }
            QPushButton#save_txt_button:pressed { background-color: #0f6674; }
            /* Disabled state inherits general QPushButton disabled style */

            /* Checkbox Style */
            QCheckBox { font-size: 10pt; color: #495057; spacing: 5px; /* Spacing between indicator and text */ }
            QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #ced4da; border-radius: 3px; background-color: white; }
            QCheckBox::indicator:checked { background-color: #007bff; border: 1px solid #0056b3; /* Blue check */ }
            QCheckBox::indicator:hover { border: 1px solid #007bff; /* Blue border on hover */ }

            /* Text Edit Style */
            QTextEdit {
                background-color: #ffffff; border: 1px solid #ced4da; border-radius: 4px;
                font-size: 10pt; color: #343a40; padding: 5px;
            }
            /* Progress Bar Style */
            QProgressBar {
                border: 1px solid #ced4da; border-radius: 4px; height: 8px; text-align: center;
                background-color: #e9ecef; /* Background of the bar */
            }
            QProgressBar::chunk { background-color: #007bff; border-radius: 4px; } /* Color of the progress indicator */

            /* Result Frame Style */
            QFrame#result_frame { border: 1px solid #dee2e6; border-radius: 5px; background-color: #ffffff; }

            /* Scroll Bar Styles (Vertical and Horizontal) */
            QScrollBar:vertical { border: none; background: #f8f9fa; width: 10px; margin: 0px 0px 0px 0px; }
            QScrollBar::handle:vertical { background: #ced4da; min-height: 20px; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #adb5bd; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
            QScrollBar:horizontal { border: none; background: #f8f9fa; height: 10px; margin: 0px 0px 0px 0px; }
            QScrollBar::handle:horizontal { background: #ced4da; min-width: 20px; border-radius: 5px; }
            QScrollBar::handle:horizontal:hover { background: #adb5bd; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; width: 0px; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
         """
         self.setStyleSheet(style_sheet) # Apply the defined styles

    # --- Slot Functions (Event Handlers) ---

    def select_file(self):
        """Opens a dialog to select an audio file and updates the UI."""
        # Suggest starting directory based on last selected file or current directory
        start_dir = str(Path(self.selected_file_path).parent) if self.selected_file_path else "."
        # Define supported file formats
        file_filter = "Audio Files (*.mp3 *.mp4 *.mpeg *.mpga *.m4a *.wav *.webm *.flac);;All Files (*)"
        # Open the file dialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", start_dir, file_filter)
        # If a file was selected
        if file_path:
            self.selected_file_path = file_path # Store the selected path
            p = Path(file_path)
            # Update the file label to show the filename
            self.file_label.setText(f"Selected: {p.name}"); self.file_label.setStyleSheet("color: #28a745; font-style: normal;") # Green text, normal style
            self.file_label.setToolTip(file_path) # Show full path on hover
            # Enable the process button, disable save buttons until processing is done
            self.process_button.setEnabled(True)
            self.save_pdf_button.setEnabled(False) # Renamed variable
            self.save_txt_button.setEnabled(False) # Disable new button
            # Reset UI elements
            self.result_display.clear(); self.update_status("File selected. Ready to process.")
            self.progress_bar.setValue(0); self.progress_bar.setVisible(False)
            self.last_successful_result = None # Clear previous results

    def update_translation_option(self, state):
        """Updates the backend processor's translation setting when the checkbox changes."""
        if hasattr(self, 'processor') and self.processor:
            is_checked = (state == Qt.CheckState.Checked.value)
            self.processor.translate_enabled = is_checked # Update backend flag
            # Check if translation is enabled but the API client isn't ready
            global client # Access the global client variable from backend
            if is_checked and not client:
                # Show a warning if trying to enable translation without a valid API key/client
                self.update_status("Warning: Translation enabled, but OpenAI client is not initialized (check API Key in backend.py).", is_error=True)
                QMessageBox.warning(self, "Translation API Warning",
                                    "Translation is enabled, but the connection to the OpenAI API could not be established.\n"
                                    "Please verify the API_KEY setting in the backend.py file.")
            # print(f"Debug: Translation option {'enabled' if is_checked else 'disabled'}") # Debug log

    def start_processing(self):
        """Initiates the audio processing task in the background thread."""
        # --- Pre-checks ---
        if not self.selected_file_path:
            self.update_status("Error: No audio file selected!", is_error=True); QMessageBox.warning(self, "No File Selected", "Please select an audio file before starting processing."); return
        if not self.processor:
            self.update_status("Error: Backend processor is not available!", is_error=True); QMessageBox.critical(self, "Critical Error", "The backend processing module could not be initialized. Cannot proceed."); return
        # Check specifically if the local Whisper model loaded successfully
        if not self.processor.local_whisper_model:
            self.update_status("Error: Local Whisper model failed to load!", is_error=True); QMessageBox.critical(self, "Model Error", "The local Whisper speech recognition model could not be loaded.\nPlease check the backend console output or Whisper installation."); return

        # --- Disable UI elements during processing ---
        self.process_button.setEnabled(False); self.select_button.setEnabled(False)
        self.save_pdf_button.setEnabled(False); self.save_txt_button.setEnabled(False) # Disable both save buttons
        self.translate_checkbox.setEnabled(False)
        # ---------------------------------------------

        # --- Reset UI for new process ---
        self.progress_bar.setValue(0); self.progress_bar.setVisible(True) # Show progress bar
        self.result_display.clear(); self.update_status("Initializing processing...")
        self.last_successful_result = None # Clear previous results

        # --- Create and Start Worker Thread ---
        self.processing_thread = Worker(self.processor, self.selected_file_path)
        # Connect signals from the worker thread to slots (handler functions) in the main window
        self.processing_thread.signals.status.connect(self.update_status)       # Update status label
        self.processing_thread.signals.progress.connect(self.update_progress)    # Update progress bar
        self.processing_thread.signals.finished.connect(self.processing_finished) # Handle successful completion
        self.processing_thread.signals.error.connect(self.handle_error)          # Handle errors
        # Ensure the thread object is deleted later to free resources
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)
        # Start the thread execution
        self.processing_thread.start()

    def update_status(self, message, is_error=False):
        """Updates the text and color of the status label."""
        self.status_label.setText(f"Status: {message}")
        # Set text color based on whether it's an error, success, warning, or info message
        if is_error: self.status_label.setStyleSheet("color: #dc3545;") # Red for errors
        elif "complete" in message.lower() or "success" in message.lower() or "saved" in message.lower():
            self.status_label.setStyleSheet("color: #28a745;") # Green for success/completion
        elif "warning" in message.lower():
             self.status_label.setStyleSheet("color: #ffc107;") # Amber for warnings
        else:
            self.status_label.setStyleSheet("color: #495057;") # Default text color
        QApplication.processEvents() # Force immediate UI update for status changes

    def update_progress(self, value):
        """Updates the value of the progress bar."""
        self.progress_bar.setValue(value)

    def processing_finished(self, success, result_data):
        """Slot executed when the background worker thread signals completion."""
        self.progress_bar.setVisible(False) # Hide progress bar
        # Re-enable general controls
        self.process_button.setEnabled(True); self.select_button.setEnabled(True); self.translate_checkbox.setEnabled(True)

        if success and result_data:
            # --- Handle Successful Processing ---
            self.update_status("Processing successfully completed!")
            self.last_successful_result = result_data # Store results for saving
            self.display_results(result_data)       # Show results in the text area
            # Enable SAVE buttons only on success with results
            self.save_pdf_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            # ------------------------------------------
        elif success and not result_data:
            # --- Handle Success but No Results ---
            self.update_status("Processing completed, but no valid segments were generated.", is_error=True) # Indicate potential issue
            self.result_display.setText("The audio was processed, but no valid text or speaker information could be extracted.\nThis might happen with silent files or very noisy audio.")
            # Keep save buttons disabled
            self.save_pdf_button.setEnabled(False)
            self.save_txt_button.setEnabled(False)
        else: # success is False (Processing failed)
             # Error message should have been set by handle_error or status updates
            if not self.status_label.text().startswith("Status: Error"): # Set generic failure if no specific error was shown
                self.update_status("Processing failed. Please check console/logs for details.", is_error=True)
            # Keep save buttons disabled
            self.save_pdf_button.setEnabled(False)
            self.save_txt_button.setEnabled(False)
        # Clean up reference to the finished thread
        self.processing_thread = None

    def handle_error(self, error_message):
        """Slot executed when the background worker signals a critical error."""
        # Extract a short version for the status bar
        short_error = error_message.splitlines()[0];
        if len(short_error) > 100: short_error = short_error[:100] + "..." # Truncate long errors
        # Update UI
        self.update_status(f"Error: {short_error}", is_error=True) # Show error in status bar
        self.result_display.setText(f"An error occurred during processing:\n\n{error_message}") # Show full error in text area
        QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{short_error}\n\nSee the result area for more details.") # Show error dialog
        # Reset UI state after error
        self.progress_bar.setVisible(False); self.process_button.setEnabled(True); self.select_button.setEnabled(True); self.translate_checkbox.setEnabled(True)
        self.save_pdf_button.setEnabled(False) # Keep save disabled
        self.save_txt_button.setEnabled(False) # Keep save disabled
        self.processing_thread = None # Clean up thread reference

    def display_results(self, result_data):
        """Formats and displays the final processed segments in the QTextEdit."""
        self.result_display.clear() # Clear previous results
        # Validate result data structure
        if not result_data or not isinstance(result_data, list):
            self.result_display.setText("Error: Could not display results due to invalid data format."); return

        formatted_text = "" # String builder for the display
        for entry in result_data:
            # Extract data safely using .get with defaults
            start_str = format_timestamp(entry.get('start')) # Format time
            end_str = format_timestamp(entry.get('end'))
            speaker = entry.get('speaker', 'Unknown') # Default speaker
            text = entry.get('text', '').strip()      # Get text, default empty
            translation = entry.get('translation')    # Get translation (can be None)

            # Append formatted entry to the string
            formatted_text += f"[{start_str} --> {end_str}] {speaker}:\n"
            formatted_text += f"  {text}\n"
            # Only add translation line if translation is enabled AND exists for this entry
            if self.translate_checkbox.isChecked() and translation:
                formatted_text += f"    Translation: {translation}\n"
            # Add a visual separator between entries
            formatted_text += "-" * 40 + "\n\n"

        # Set the text in the display area
        if not formatted_text:
            # Message if processing succeeded but filtering removed all segments
            self.result_display.setText("Processing completed, but no valid segments remained after filtering.")
        else:
            self.result_display.setText(formatted_text.strip()) # Remove trailing newlines
        # Scroll to the top of the result display
        self.result_display.verticalScrollBar().setValue(0)

    def _get_default_save_path(self, extension):
        """Helper function to generate a default filename and path for saving."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_file_stem = "transcript" # Default stem
        # Try to base the name on the input file if available
        if self.selected_file_path:
            original_file_stem = Path(self.selected_file_path).stem
        # Construct filename
        default_filename = f"{original_file_stem}_{timestamp}.{extension}"
        # Use the configured output directory
        default_save_dir = Path(OUTPUT_DIR)
        try:
             # Ensure directory exists, create if not
             default_save_dir.mkdir(parents=True, exist_ok=True)
             # Return the full suggested path
             return str(default_save_dir.resolve() / default_filename)
        except Exception as e:
             # Fallback to current directory if default output dir is problematic
             print(f"Warning: Could not use default output directory '{OUTPUT_DIR}' ({e}). Suggesting save to current directory.");
             return default_filename # Return filename only

    def save_pdf(self):
        """Handles the 'Save PDF' button click: opens dialog, calls backend."""
        # Check if there are results to save
        if not self.last_successful_result:
            self.update_status("Error: No processing results available to save.", is_error=True); QMessageBox.warning(self, "Cannot Save PDF", "There are no results from the last successful processing run to save."); return
        if not self.processor:
            self.update_status("Error: Backend processor is unavailable.", is_error=True); QMessageBox.critical(self, "Save Error", "Cannot save PDF because the backend processor is not initialized."); return

        # Get default save path
        default_save_path = self._get_default_save_path("pdf")
        # Open "Save As" dialog
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result as PDF", # Dialog title
            default_save_path,    # Suggested path/filename
            "PDF Files (*.pdf)"   # File type filter
        )

        # If the user selected a path (didn't cancel)
        if save_path:
            # QFileDialog usually handles the extension based on filter, but double-check/add if needed
            # if not save_path.lower().endswith(".pdf"): save_path += ".pdf"
            self.update_status(f"Saving results to PDF: {Path(save_path).name}...")
            QApplication.processEvents() # Update UI to show status
            try:
                # Call the backend's save_to_pdf method with the chosen path
                self.processor.save_to_pdf(output_filename=save_path)
                # Success message handled by the backend's _update_status
                QMessageBox.information(self, "Save Successful", f"Results successfully saved as PDF to:\n{save_path}")
            except Exception as e:
                # Handle errors during backend save operation
                error_msg = f"Error occurred while saving PDF file: {e}\n{traceback.format_exc()}"
                self.update_status(f"Error: Failed to save PDF.", is_error=True)
                QMessageBox.critical(self, "Save Failed", f"An error occurred while saving the PDF file:\n{e}")
                print(error_msg) # Log detailed error

    # *** NEW SLOT: save_txt ***
    def save_txt(self):
        """Handles the 'Save TXT' button click: opens dialog, calls backend."""
        # Check if there are results to save
        if not self.last_successful_result:
            self.update_status("Error: No processing results available to save.", is_error=True); QMessageBox.warning(self, "Cannot Save TXT", "There are no results from the last successful processing run to save."); return
        if not self.processor:
            self.update_status("Error: Backend processor is unavailable.", is_error=True); QMessageBox.critical(self, "Save Error", "Cannot save TXT because the backend processor is not initialized."); return

        # Get default save path with .txt extension
        default_save_path = self._get_default_save_path("txt")
        # Open "Save As" dialog for TXT files
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result as TXT",      # Dialog title
            default_save_path,         # Suggested path/filename
            "Text Files (*.txt);;All Files (*)" # Filter for .txt
        )

        # If the user selected a path
        if save_path:
            # Ensure the file has a .txt extension, especially if "All Files" was selected
            if "." not in Path(save_path).name: # Basic check if extension is missing
                 save_path += ".txt"
            elif not save_path.lower().endswith(".txt"):
                 # Handle cases where user might type a different extension
                 # Decide whether to force .txt or allow other extensions
                 # Forcing .txt might be safer:
                 save_path = str(Path(save_path).with_suffix('.txt'))


            self.update_status(f"Saving results to TXT: {Path(save_path).name}...")
            QApplication.processEvents() # Update UI
            try:
                # Call the NEW backend save_to_txt method
                self.processor.save_to_txt(output_filename=save_path)
                # Success message handled by backend's _update_status
                QMessageBox.information(self, "Save Successful", f"Results successfully saved as TXT to:\n{save_path}")
            except Exception as e:
                # Handle errors during backend TXT save operation
                error_msg = f"Error occurred while saving TXT file: {e}\n{traceback.format_exc()}"
                self.update_status(f"Error: Failed to save TXT.", is_error=True)
                QMessageBox.critical(self, "Save Failed", f"An error occurred while saving the TXT file:\n{e}")
                print(error_msg) # Log detailed error


# --- Application Entry Point ---
if __name__ == "__main__":
    # Check if backend is available before even creating the QApplication
    if not BACKEND_AVAILABLE:
         # Need a temporary QApplication instance to show the message box if backend failed
         app_temp = QApplication(sys.argv)
         QMessageBox.critical(None, "Application Startup Error",
                              "Failed to load critical backend components (backend.py).\n"
                              "Please ensure the file exists and all dependencies are installed.\n"
                              "The application cannot start.")
         sys.exit(1) # Exit if backend is unusable

    # Create the main application instance
    app = QApplication(sys.argv)
    # Set application metadata (optional)
    app.setApplicationName("IntelliTranscribe")
    app.setOrganizationName("YourOrganization") # Replace as needed

    # Set global font (optional, affects all widgets unless overridden)
    # try:
    #     font = QFont("Segoe UI", 10) # Example: Use Segoe UI
    #     app.setFont(font)
    # except Exception as font_e:
    #     print(f"Warning: Could not set global font: {font_e}")

    # Create the main window instance
    window = MainWindow()

    # Only show the window and run the app if initialization was successful
    # Check if window object was created AND backend is available AND processor was initialized
    if window and BACKEND_AVAILABLE and window.processor:
         window.show() # Display the main window
         sys.exit(app.exec()) # Start the Qt event loop
    else:
         # If initialization failed (e.g., critical error shown in __init__), exit gracefully
         print("Application initialization failed. Exiting.")
         # Ensure error message was shown before exiting
         if not BACKEND_AVAILABLE or not (hasattr(window, 'processor') and window.processor):
              if 'app_temp' not in locals(): app_temp = QApplication(sys.argv) # Ensure app exists for message box
              QMessageBox.critical(None,"Application Startup Error", "Failed during initialization. Please check console logs for errors.")
         sys.exit(1)