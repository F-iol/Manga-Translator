import sys
import os
import platform
# making sure main.py opens the app
def restart_in_venv():
    VENV_FOLDER_NAME = "venv" 
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if platform.system() == "Windows":
        venv_executable = os.path.join(base_dir, VENV_FOLDER_NAME, "Scripts", "python.exe")
    else:
        venv_executable = os.path.join(base_dir, VENV_FOLDER_NAME, "bin", "python")


    if os.path.exists(venv_executable):
        current_exe = os.path.normpath(sys.executable)
        target_exe = os.path.normpath(venv_executable)
        
        if current_exe != target_exe:

            os.execv(venv_executable, [venv_executable] + sys.argv)
restart_in_venv()

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QStatusBar, QSizePolicy, QStackedWidget, QSplitter,
    QLineEdit, QGridLayout, QFrame, QTextEdit, QCheckBox ) 
from PySide6.QtCore import Qt, QObject, Signal, QByteArray, QBuffer, QIODevice, QSize
from PySide6.QtGui import QPixmap
from ultralytics import YOLO
from manga_ocr import MangaOcr
from transformers import MarianMTModel, MarianTokenizer
import torch
import time
from bubble_logic import BubbleTranslatorManager 
from snipper_logic import get_snipping_manager 
import warnings
warnings.filterwarnings('ignore')

class TranslationSignals(QObject):
    #signals for files
    new_output = Signal(str)
    new_image_data = Signal(bytes)
    hotkey_triggered = Signal() 
    
KEY_MAP = {
    Qt.Key.Key_Control: "Control",
    Qt.Key.Key_Shift: "Shift",
    Qt.Key.Key_Alt: "Alt",
    Qt.Key.Key_A: "A",
    Qt.Key.Key_B: "B",
    Qt.Key.Key_C: "C",
    Qt.Key.Key_D: "D",
    Qt.Key.Key_Q: "Q",
    Qt.Key.Key_S: "S",
    Qt.Key.Key_P: "P",
    Qt.Key.Key_E: "E",
    Qt.Key.Key_E: "H",
    Qt.Key.Key_E: "G",
    Qt.Key.Key_E: "I",
    Qt.Key.Key_E: "T",
    Qt.Key.Key_E: "N",
    Qt.Key.Key_E: "M",
    Qt.Key.Key_E: "X",
    Qt.Key.Key_E: "Z",
    Qt.Key.Key_E: "V",
    Qt.Key.Key_E: "W",
    Qt.Key.Key_E: "Y",
    Qt.Key.Key_E: "U",            
    Qt.Key.Key_Escape: "Escape",
}
MODIFIERS = ["Control", "Shift", "Alt"]
MODELS_AVAILABLE = True

def load_models(): 
    models ={} 
    try:
        DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
        CUSTOM_FONT_PATH = './fonts/PermanentMarker-Regular.ttf'
        TRANSLATION_MODEL_NAME = 'Helsinki-NLP/opus-mt-ja-en'
        TRANSLATION_MODEL_PATH = './models'
        BUBBLE_PATH = './models/bubble_model.pt'

        os.environ['TRANSFORMERS_CACHE'] = TRANSLATION_MODEL_PATH
        os.environ['HF_HOME'] = TRANSLATION_MODEL_PATH
        models['ocr'] = MangaOcr()
        models['bubble'] = YOLO(BUBBLE_PATH)
        models['tokenizer'] = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME, cache_dir=TRANSLATION_MODEL_PATH)
        models['translator'] = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME, cache_dir=TRANSLATION_MODEL_PATH).to(DEVICE)
        models['device'] = DEVICE
        return models
    except Exception as e:
        print(f'Models not loaded {e}')
        return {}
    
def format_combination_for_display(keys_list: list) -> str:
    return ' + '.join(keys_list)

class MainView(QWidget):
    #main window
    def __init__(self, signals, translator_manager):
        super().__init__()
        self.setObjectName("MainView")
        self.signals = signals
        self.translator_manager = translator_manager

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("MainSplitter")

        self.image_label = QLabel("Translator Initializing... Press Shift + E to select an area and start the Bubble Translator.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("CentralContentBox")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False) 
        self.image_label.setMinimumSize(1, 1) 
        self.image_label.setFrameShape(QFrame.StyledPanel)
        self.image_label.setWordWrap(True)

        self.console_widget = QTextEdit()
        self.console_widget.setReadOnly(True)
        self.console_widget.setText(
            "Console Output Here...\n\n"
            "Controls:\n"
            "- Press Shift + E to select an area and start continuous translation.\n"
            "- Press Shift + S or ESC to stop the continuous translation loop.\n"
            "- Press Shift + Q to select area and get single translation. \n"
        )
        self.console_widget.setObjectName("ConsoleWidget")
        self.console_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.console_widget.setMinimumSize(1, 1) 

        splitter.addWidget(self.image_label)
        splitter.addWidget(self.console_widget)
        splitter.setSizes([700, 300])
        splitter.setHandleWidth(5)

        main_layout.addWidget(splitter)

        self.signals.new_output.connect(self.append_console_output)
        self.signals.new_image_data.connect(self.display_translated_image)
        self.signals.hotkey_triggered.connect(self.translator_manager.start_continuous_translation)
        
        self.translator_manager.set_gui_callbacks(
            output_callback=self.signals.new_output.emit,
            image_callback=self.signals.new_image_data.emit,
            hotkey_callback=self.signals.hotkey_triggered.emit 
        )

    def append_console_output(self, text):
        current_text = self.console_widget.toPlainText()
        new_text = f"[{time.strftime('%H:%M:%S')}] "+"\n" + text + "\n\n" + current_text
        self.console_widget.setText(new_text.strip())

    def display_translated_image(self, image_data: bytes):
        pixmap = QPixmap()
        byte_array = QByteArray(image_data)
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.ReadOnly)
        
        if pixmap.loadFromData(buffer.data()):
            label_size = self.image_label.size()
            should_scale = (
                pixmap.width() > label_size.width() or 
                pixmap.height() > label_size.height()
            )
            
            if should_scale:
                final_pixmap = pixmap.scaled(
                    label_size, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation 
                )
            else:
                final_pixmap = pixmap

            self.image_label.setPixmap(final_pixmap) 
            self.image_label.setAlignment(Qt.AlignCenter)
        else:
            self.image_label.setText("Error displaying image.")
            self.append_console_output("ERROR: Failed to load image data into QPixmap.")

class SettingsView(QWidget):
    #inference
    def __init__(self, snipper_manager, bubble_translator_manager):
        super().__init__()
        self.setObjectName("SettingsView")
        self.snipper_manager = snipper_manager
        self.bubble_manager = bubble_translator_manager

        settings_layout = QVBoxLayout(self)
        settings_layout.setContentsMargins(15, 15, 15, 15)

        self.settings_box_content = QFrame()
        self.settings_box_content.setObjectName("SettingsBox")
        settings_box_layout = QVBoxLayout(self.settings_box_content)

        settings_box_layout.addWidget(QLabel("<h2>Application Settings</h2>"))
        settings_box_layout.addSpacing(20)

        settings_box_layout.addWidget(QLabel("<h3>Snipping Tool Hotkey:</h3>"))
        settings_box_layout.addSpacing(5)
        self.setup_shortcut_group(
            manager=self.snipper_manager,
            layout=settings_box_layout,
            label_text="Snipping Shortcut:",
            attribute_prefix="snipping",
            get_combo_func=lambda: self.snipper_manager.combination,
            set_combo_func=self.snipper_manager.set_combination
        )
        settings_box_layout.addSpacing(30)
        
        settings_box_layout.addWidget(QLabel("<h3>Live Translation Hotkeys:</h3>"))
        settings_box_layout.addSpacing(5)
        
        self.setup_shortcut_group(
            manager=self.bubble_manager,
            layout=settings_box_layout,
            label_text="Start Translation:",
            attribute_prefix="start_live",
            get_combo_func=self.bubble_manager.get_start_combination,
            set_combo_func=self.bubble_manager.set_start_combination
        )
        settings_box_layout.addSpacing(15)

        self.setup_shortcut_group(
            manager=self.bubble_manager,
            layout=settings_box_layout,
            label_text="Stop Translation (Combo):",
            attribute_prefix="stop_live",
            get_combo_func=self.bubble_manager.get_stop_combination,
            set_combo_func=self.bubble_manager.set_stop_combination
        )
        settings_box_layout.addSpacing(15)
        
        self.setup_shortcut_group(
            manager=self.bubble_manager,
            layout=settings_box_layout,
            label_text="Stop Translation (Single Key):",
            attribute_prefix="stop_live_v2",
            get_combo_func=self.bubble_manager.get_stop_v2_combination,
            set_combo_func=self.bubble_manager.set_stop_v2_combination
        )
        settings_box_layout.addSpacing(30)

        output_group = QFrame()
        output_group.setLayout(QVBoxLayout())
        output_group.layout().setContentsMargins(0, 0, 0, 0)
        output_group.setObjectName("SettingGroupFrame")

        output_title = QLabel("<h3>Console Output Preferences:</h3>")
        output_title.setStyleSheet("color: #F4A460; font-size: 14px; margin-bottom: 5px;")
        output_group.layout().addWidget(output_title)

        self.original_check = QCheckBox("Display Original (Untranslated) Text")
        initial_original_state = Qt.Checked if self.snipper_manager.get_display_original() else Qt.Unchecked
        self.original_check.setCheckState(initial_original_state)
        self.original_check.stateChanged.connect(self.snipper_manager.set_display_original_from_qt)
        output_group.layout().addWidget(self.original_check)

        self.translated_check = QCheckBox("Display Translated Text")
        initial_translated_state = Qt.Checked if self.snipper_manager.get_display_translated() else Qt.Unchecked
        self.translated_check.setCheckState(initial_translated_state)
        self.translated_check.stateChanged.connect(self.snipper_manager.set_display_translated_from_qt)
        output_group.layout().addWidget(self.translated_check)

        self.image_check = QCheckBox("Display Translated Image")
        initial_image_state = Qt.Checked if self.snipper_manager.get_display_image() else Qt.Unchecked
        self.image_check.setCheckState(initial_image_state)
        self.image_check.stateChanged.connect(self.snipper_manager.set_display_image_from_qt)
        output_group.layout().addWidget(self.image_check)

        settings_box_layout.addWidget(output_group)
        settings_box_layout.addStretch()

        settings_layout.addWidget(self.settings_box_content)

        self.is_capturing = False
        self.temp_keys = set()
        self.active_display = None 
        self.active_setter = None 
        self.active_getter = None 
        self.last_valid_keys = []
        self._update_last_valid_keys(self.snipper_manager.combination)

    def setup_shortcut_group(self, manager, layout, label_text, attribute_prefix, get_combo_func, set_combo_func):
        # ui shortcut for single translation capure
        def get_initial_display():
            try:
                key_strings = get_combo_func()
                if not key_strings: return "Not configured"
                return format_combination_for_display(key_strings)
            except Exception:
                return "Shift + Q"
        
        shortcut_group = QWidget()
        shortcut_layout = QGridLayout(shortcut_group)
        shortcut_layout.setColumnStretch(1, 1)

        shortcut_display = QLineEdit(get_initial_display())
        shortcut_display.setReadOnly(True)
        shortcut_display.setObjectName("ShortcutDisplay")
        
        change_btn = QPushButton("Change Shortcut")
        change_btn.clicked.connect(
            lambda checked, disp=shortcut_display, setter=set_combo_func, getter=get_combo_func: 
            self.start_shortcut_capture(disp, setter, getter)
        )
        
        shortcut_layout.addWidget(QLabel(label_text), 0, 0, Qt.AlignLeft)
        shortcut_layout.addWidget(shortcut_display, 0, 1)
        shortcut_layout.addWidget(change_btn, 0, 2)
        layout.addWidget(shortcut_group)

    def start_shortcut_capture(self, display_widget, setter_func, getter_func):
        if self.is_capturing: return
        self.is_capturing = True
        self.temp_keys = set()
        self.active_display = display_widget
        self.active_setter = setter_func
        self.active_getter = getter_func
        
        try:
             self.last_valid_keys = getter_func() 
        except:
             self.last_valid_keys = ["Control", "Q"] 

        self.active_display.setText("Press your new combination (Esc to cancel)...")
        self.active_display.setStyleSheet("background-color: #582C00;")
        
        parent_widget = display_widget.parent()
        change_btn = parent_widget.findChild(QPushButton)
        if change_btn:
            change_btn.setText("Capturing...")
            change_btn.setEnabled(False)
        self.setFocus(Qt.ActiveWindowFocusReason)

    def keyPressEvent(self, event):
        if not self.is_capturing:
            super().keyPressEvent(event)
            return

        key_code = event.key()
        key_name = None 

        if key_code == Qt.Key.Key_Escape:
            is_setting_single_key = len(self.active_getter()) == 1 if self.active_getter else False
            if is_setting_single_key:
                key_name = KEY_MAP.get(key_code)
            else:
                self.reset_ui(self.last_valid_keys)
                event.accept()
                return

        if key_code in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return

        if not key_name:
            key_name = KEY_MAP.get(key_code)

        if not key_name and event.text() and event.text().isalnum():
            key_name = event.text().upper()

        if key_name and key_name != "Escape":
            if key_name not in self.temp_keys:
                self.temp_keys.add(key_name)
            
            sorted_keys = sorted(list(self.temp_keys), key=lambda k: (k not in MODIFIERS, k))
            self.active_display.setText(format_combination_for_display(sorted_keys))

            if key_name not in MODIFIERS:
                self._finish_shortcut_capture_internal(sorted_keys)
        
        elif key_name == "Escape":
            self.temp_keys.add(key_name)
            self._finish_shortcut_capture_internal([key_name])

        event.accept()
        
    def _update_last_valid_keys(self, combo_list_of_strings):
        self.last_valid_keys = combo_list_of_strings
        
    def _finish_shortcut_capture_internal(self, captured_keys):
        #validate and apply recorded keys
        has_modifier = any(k in MODIFIERS for k in captured_keys)
        has_action_key = any(k not in MODIFIERS for k in captured_keys)
        is_single_key = len(captured_keys) == 1
        is_setting_single_key = len(self.active_getter()) == 1

        if is_setting_single_key:
              if not is_single_key:
                self.reset_ui(self.last_valid_keys, error_message="Single-key combo requires exactly one key.")
                return
        elif len(captured_keys) < 2 or not has_modifier or not has_action_key:
            self.reset_ui(self.last_valid_keys, error_message="Combo requires a Modifier (Ctrl/Shift/Alt) AND an Action Key.")
            return

        self.active_setter(captured_keys) 
        self.last_valid_keys = captured_keys 
        self.reset_ui(captured_keys)

    def reset_ui(self, final_keys, error_message=None):
        self.is_capturing = False
        self.temp_keys.clear()
        
        if not self.active_display:
            return

        parent_widget = self.active_display.parent()
        change_btn = parent_widget.findChild(QPushButton)
        
        if change_btn:
            change_btn.setEnabled(True)
            change_btn.setText("Change Shortcut")
            
        self.active_display.setReadOnly(True)

        if error_message:
            self.active_display.setText(error_message)
            self.active_display.setStyleSheet("background-color: #7B0000;")
        else:
            self.active_display.setText(format_combination_for_display(final_keys))
            self.active_display.setStyleSheet("")
            
        self.active_display = None
        self.active_setter = None
        self.active_getter = None
        self.clearFocus()

class ModernWindow(QMainWindow):
    #Main window integrating all views and managers
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Translator")
        self.resize(800, 600) 

        self.signals = TranslationSignals()
        self.snipper_manager = get_snipping_manager()
        self.bubble_translator_manager = BubbleTranslatorManager()
        load_models_dict = load_models()
        self.bubble_translator_manager.set_models(load_models_dict)
        self.snipper_manager.set_models(load_models_dict)
        self.snipper_manager.set_gui_output_callback(self.signals.new_output.emit)
        self.setStyleSheet(self.get_stylesheet())

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)

        self.view_stack = QStackedWidget()
        self.view_stack.setObjectName("ViewStack")
        main_layout.addWidget(self.view_stack, 1)

        self.main_view = MainView(self.signals, self.bubble_translator_manager) 
        self.settings_view = SettingsView(self.snipper_manager,self.bubble_translator_manager)

        self.view_stack.addWidget(self.main_view)   
        self.view_stack.addWidget(self.settings_view)

        self.setStatusBar(self.create_status_bar())

        self.main_btn.clicked.connect(lambda: self.switch_view(0, self.main_btn))
        self.settings_btn.clicked.connect(lambda: self.switch_view(1, self.settings_btn))

        self.main_btn.setChecked(True)
        self.view_stack.setCurrentIndex(0)

        self.bubble_translator_manager.start_hotkey_listener()
        QApplication.instance().aboutToQuit.connect(self._cleanup)

    def _cleanup(self):
        """Stops background listeners on exit."""
        if self.snipper_manager:
            self.snipper_manager.stop_listeners()
        if self.bubble_translator_manager:
            self.bubble_translator_manager.stop_listeners()

    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setFixedWidth(70)
        sidebar.setObjectName("Sidebar")

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 20, 0, 10)
        sidebar_layout.setSpacing(5)

        title_label = QLabel("Manga\nTranslator")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("SidebarTitle")
        sidebar_layout.addWidget(title_label)

        self.main_btn = QPushButton("🏠\nMain")
        self.settings_btn = QPushButton("⚙️\nSettings")

        for btn in [self.main_btn, self.settings_btn]:
            btn.setCheckable(True)
            btn.setMinimumHeight(60)
            btn.setStyleSheet("text-align: center;")

        sidebar_layout.addWidget(self.main_btn)
        sidebar_layout.addWidget(self.settings_btn)
        sidebar_layout.addStretch()
        return sidebar

    def create_status_bar(self):
        status_bar = QStatusBar()
        status_bar.setObjectName("StatusBar")
        status_bar.addWidget(QLabel("© Manga Translator "))
        github_label = QLabel('Github : <a href="https://github.com/F-iol">F-iol GitHub</a>')
        github_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        github_label.setOpenExternalLinks(True)
        status_bar.addPermanentWidget(github_label)
        return status_bar

    def switch_view(self, index, sender_btn):
        if index != 1 and self.view_stack.currentWidget() == self.settings_view and self.settings_view.is_capturing:
             self.settings_view.reset_ui(self.settings_view.last_valid_keys)

        self.view_stack.setCurrentIndex(index)
        for btn in [self.main_btn, self.settings_btn]:
            btn.setChecked(btn == sender_btn)

    def get_stylesheet(self):
        ACCENT_COLOR = "#F4A460"
        return f"""
        QMainWindow {{ background-color: #1E1E1E; }}
        QLabel, QCheckBox {{ color: white; }}
        QLabel#CentralContentBox {{
            color: #AAAAAA; 
            font-size: 16px; 
            font-style: italic;
        }}
        QLabel > h2 {{
            font-size: 18px;
            color: {ACCENT_COLOR};
            font-weight: bold;
        }}
        QCheckBox::indicator {{
            width: 13px;
            height: 13px;
            border: 1px solid #555555;
            border-radius: 3px;
            background-color: #2D2D2D;
        }}
        QCheckBox::indicator:checked {{
            background-color: {ACCENT_COLOR};
            border: 1px solid {ACCENT_COLOR};
        }}
        #Sidebar {{
            background-color: #2D2D2D;
            border-right: 2px solid {ACCENT_COLOR};
        }}
        #SidebarTitle {{
            font-size: 12px;
            font-weight: bold;
            color: {ACCENT_COLOR};
            padding: 10px 0;
            margin-bottom: 10px;
        }}
        QPushButton {{
            color: white;
            background-color: transparent;
            border: none;
            padding: 5px 0;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: #3C3C3C;
            margin: 0 5px;
        }}
        QPushButton:checked {{
            background-color: {ACCENT_COLOR};
            font-weight: bold;
            margin: 0 5px;
        }}
        #CentralContentBox, #SettingsBox {{
            background-color: #1A1A1A;
            border-radius: 10px;
            border: 1px solid #555555;
            padding: 20px;
        }}
        QLineEdit {{
            background-color: #252525;
            border: 1px solid #555555;
            border-radius: 5px;
            padding: 5px 10px;
            color: white;
        }}
        QLineEdit#ShortcutDisplay[readOnly="true"] {{
            color: {ACCENT_COLOR};
            font-weight: bold;
        }}
        #ConsoleWidget {{
            background-color: #252525;
            min-height: 500px;
            font-family: monospace;
            color: #AAAAAA;
            font-size: 11px;
            border-radius: 10px;
            border: 1px solid #555555;
        }}
        #StatusBar {{
            background-color: #1E1E1E;
            color: #AAAAAA;
            border-top: 1px solid #444444;
        }}
        QSplitter::handle {{ background: #3C3C3C; }}
        QSplitter::handle:horizontal {{
            width: 5px;
            margin-left: 0px;
            margin-right: 0px;
            border-radius: 2px;
            border: 1px solid #444444;
        }}
        QSplitter::handle:hover {{ background: {ACCENT_COLOR}; }}
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernWindow()
    window.show()
    sys.exit(app.exec())