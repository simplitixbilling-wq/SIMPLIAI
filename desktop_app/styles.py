"""QSS themes — Modern deep-navy + indigo accent for PySide6."""

DARK_THEME = """
/* ─── Global ─────────────────────────────────────────── */
QWidget {
    background-color: #0d1017;
    color: #e2e4ec;
    font-family: "Segoe UI Variable", "Segoe UI", system-ui, sans-serif;
    font-size: 13px;
}
QMainWindow { background-color: #0d1017; }

/* ─── Top Bar ────────────────────────────────────────── */
#TopBar {
    background-color: #0d1017;
    border-bottom: 1px solid #1c1f2e;
    min-height: 44px;
    max-height: 44px;
}
#BrandLabel {
    color: #818cf8;
    font-size: 15px;
    font-weight: 700;
    background: transparent;
}

/* ─── Sidebar ────────────────────────────────────────── */
#Sidebar {
    background-color: #111318;
    border-right: 1px solid #1c1f2e;
}
#SidebarSection {
    background: transparent;
}
#SectionHeader {
    background: transparent;
}
#SectionTitle {
    color: #6b7090;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: transparent;
}

/* ─── Buttons ────────────────────────────────────────── */
QPushButton {
    background-color: #1a1d28;
    color: #c8cad4;
    border: 1px solid #262a3a;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #222636;
    border-color: #6366f1;
    color: #e2e4ec;
}
QPushButton:pressed {
    background-color: #2a2e42;
}
QPushButton:disabled {
    background-color: #111318;
    color: #3a3e52;
    border-color: #1c1f2e;
}
QPushButton#AccentBtn {
    background-color: #6366f1;
    color: #ffffff;
    border: none;
    font-weight: 600;
    border-radius: 8px;
}
QPushButton#AccentBtn:hover {
    background-color: #818cf8;
}
QPushButton#AccentBtn:pressed {
    background-color: #5558e0;
}
QPushButton#GreenBtn {
    background-color: #22c55e;
    color: #ffffff;
    border: none;
    font-weight: 600;
    border-radius: 8px;
}
QPushButton#GreenBtn:hover {
    background-color: #4ade80;
}
QPushButton#RedBtn {
    background-color: #ef4444;
    color: #ffffff;
    border: none;
    font-weight: 600;
    border-radius: 8px;
}
QPushButton#RedBtn:hover {
    background-color: #f87171;
}
QPushButton#IconBtn {
    background: transparent;
    border: none;
    font-size: 15px;
    padding: 4px 8px;
    border-radius: 6px;
    color: #6b7090;
}
QPushButton#IconBtn:hover {
    background-color: #1a1d28;
    color: #e2e4ec;
}
QPushButton#ToolbarBtn {
    background: transparent;
    border: 1px solid #262a3a;
    border-radius: 8px;
    padding: 5px 12px;
    color: #8b90a5;
    font-size: 12px;
}
QPushButton#ToolbarBtn:hover {
    background-color: #1a1d28;
    border-color: #6366f1;
    color: #e2e4ec;
}
QPushButton#GhostBtn {
    background: transparent;
    border: 1px solid #262a3a;
    border-radius: 8px;
    padding: 5px 14px;
    color: #8b90a5;
    font-size: 12px;
}
QPushButton#GhostBtn:hover {
    background-color: #1a1d28;
    color: #e2e4ec;
    border-color: #363a50;
}
QPushButton#TinyBtn {
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 2px 4px;
    font-size: 13px;
    color: #555a70;
}
QPushButton#TinyBtn:hover {
    background-color: #1a1d28;
    color: #e2e4ec;
}
QPushButton#InputIcon {
    background-color: #151820;
    border: 1px solid #262a3a;
    border-radius: 17px;
    color: #9aa3c4;
    font-size: 16px;
    font-weight: 700;
}
QPushButton#InputIcon:hover {
    background-color: #1f2433;
    border-color: #363a50;
    color: #e2e4ec;
}
QPushButton#SendBtn {
    background-color: #6366f1;
    color: #ffffff;
    border: none;
    border-radius: 19px;
    font-size: 15px;
    font-weight: bold;
}
QPushButton#SendBtn:hover {
    background-color: #818cf8;
}
QPushButton#SendBtn:disabled {
    background-color: #1a1d28;
    color: #3a3e52;
}
QPushButton#StopBtn {
    background-color: #ef4444;
    color: #ffffff;
    border: none;
    border-radius: 17px;
    font-size: 12px;
}
QPushButton#StopBtn:hover {
    background-color: #f87171;
}
QPushButton#StopBtn:disabled {
    background-color: transparent;
    color: #3a3e52;
    border: 1px solid #1c1f2e;
}
QPushButton#ActionBtn {
    background: transparent;
    border: none;
    color: #4a4e65;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
}
QPushButton#ActionBtn:hover {
    color: #c8cad4;
    background-color: #1a1d28;
}

/* ─── ComboBox ───────────────────────────────────────── */
QComboBox {
    background-color: #1a1d28;
    color: #e2e4ec;
    border: 1px solid #262a3a;
    border-radius: 8px;
    padding: 5px 12px;
    min-height: 26px;
}
QComboBox:hover {
    border-color: #6366f1;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #6366f1;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1a1d28;
    color: #e2e4ec;
    border: 1px solid #262a3a;
    selection-background-color: #262a3a;
    selection-color: #818cf8;
    border-radius: 8px;
    padding: 4px;
}

/* ─── Line Edit ──────────────────────────────────────── */
QLineEdit {
    background-color: #1a1d28;
    color: #e2e4ec;
    border: 1px solid #262a3a;
    border-radius: 8px;
    padding: 6px 12px;
    selection-background-color: #6366f1;
    selection-color: #ffffff;
}
QLineEdit:focus {
    border-color: #6366f1;
}
QLineEdit::placeholder {
    color: #4a4e65;
}
#ChatInput {
    background-color: #151820;
    border: 1px solid #262a3a;
    border-radius: 20px;
    padding: 8px 16px;
    font-size: 13px;
}
#ChatInput:focus {
    border-color: #6366f1;
}

/* ─── Text Edit ──────────────────────────────────────── */
QTextEdit, QPlainTextEdit {
    background-color: #0d1017;
    color: #e2e4ec;
    border: 1px solid #1c1f2e;
    border-radius: 8px;
    padding: 8px;
    selection-background-color: #6366f1;
    selection-color: #ffffff;
}
QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #6366f1;
}

/* ─── Scroll Area ────────────────────────────────────── */
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}

/* ─── Scroll Bar ─────────────────────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #2a2e42;
    min-height: 40px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: #3a3e55;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 6px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #2a2e42;
    min-width: 40px;
    border-radius: 3px;
}
QScrollBar::handle:horizontal:hover {
    background: #3a3e55;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ─── Progress Bar ───────────────────────────────────── */
QProgressBar {
    background-color: #1c1f2e;
    border: none;
    border-radius: 3px;
    max-height: 6px;
    text-align: center;
    font-size: 0px;
}
QProgressBar::chunk {
    background-color: #6366f1;
    border-radius: 3px;
}

/* ─── Slider ─────────────────────────────────────────── */
QSlider::groove:horizontal {
    background: #1c1f2e;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #6366f1;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #818cf8;
}
QSlider::sub-page:horizontal {
    background: #6366f1;
    border-radius: 2px;
}

/* ─── Radio Button ───────────────────────────────────── */
QRadioButton {
    color: #e2e4ec;
    spacing: 8px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid #3a3e52;
    background-color: #1a1d28;
}
QRadioButton::indicator:checked {
    border-color: #6366f1;
    background-color: #6366f1;
}

/* ─── Labels ─────────────────────────────────────────── */
QLabel {
    color: #e2e4ec;
    background: transparent;
}
QLabel#Muted {
    color: #4a4e65;
    font-size: 11px;
}
QLabel#Accent {
    color: #818cf8;
}
QLabel#Gold {
    color: #fbbf24;
}
QLabel#StatusLabel {
    color: #8b90a5;
    font-size: 11px;
}

/* ─── Chat Area ──────────────────────────────────────── */
#ChatArea {
    background-color: #0d1017;
    border: none;
}
#ChatScroll {
    background-color: #0d1017;
}

/* ─── Chat Bubbles ───────────────────────────────────── */
#UserBubble {
    background-color: #6366f1;
    border-radius: 16px;
    border: none;
}
#UserBubble QTextEdit {
    background: transparent;
    color: #ffffff;
    border: none;
    font-size: 13px;
}
#AssistantBubble {
    background-color: #151820;
    border-radius: 16px;
    border: 1px solid #1c1f2e;
}
#AssistantBubble QTextEdit {
    background: transparent;
    color: #e2e4ec;
    border: none;
    font-size: 13px;
}

/* ─── Input Bar ──────────────────────────────────────── */
#InputBar {
    background-color: #0d1017;
    border-top: 1px solid #1c1f2e;
}

/* ─── Chat List ──────────────────────────────────────── */
#ChatRow {
    background-color: transparent;
    border-radius: 8px;
    border: none;
}
#ChatRow:hover {
    background-color: #1a1d28;
}

/* Active chat row */
#ChatRowActive {
    background-color: #1a1d28;
    border-radius: 8px;
    border: 1px solid #262a3a;
}

/* ─── RAG Rows ───────────────────────────────────────── */
#RagRow {
    background-color: transparent;
    border-radius: 6px;
    border: none;
}
#RagRow:hover {
    background-color: #1a1d28;
}

/* ─── Dialog ─────────────────────────────────────────── */
QDialog {
    background-color: #111318;
}

/* ─── Splitter ───────────────────────────────────────── */
QSplitter::handle {
    background-color: #1c1f2e;
    width: 1px;
}

/* ─── ToolTip ────────────────────────────────────────── */
QToolTip {
    background-color: #1a1d28;
    color: #e2e4ec;
    border: 1px solid #262a3a;
    border-radius: 6px;
    padding: 4px 8px;
}

/* ─── Menu ───────────────────────────────────────────── */
QMenu {
    background-color: #1a1d28;
    color: #e2e4ec;
    border: 1px solid #262a3a;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 20px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #262a3a;
    color: #818cf8;
}
QMenu::separator {
    height: 1px;
    background: #1c1f2e;
    margin: 4px 8px;
}
"""

# ══════════════════════════════════════════════════════════════════════

LIGHT_THEME = """
/* ─── Global ─────────────────────────────────────────── */
QWidget {
    background-color: #fafbfd;
    color: #1a1c24;
    font-family: "Segoe UI Variable", "Segoe UI", system-ui, sans-serif;
    font-size: 13px;
}
QMainWindow { background-color: #fafbfd; }

/* ─── Top Bar ────────────────────────────────────────── */
#TopBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e8eaef;
    min-height: 44px;
    max-height: 44px;
}
#BrandLabel {
    color: #6366f1;
    font-size: 15px;
    font-weight: 700;
    background: transparent;
}

/* ─── Sidebar ────────────────────────────────────────── */
#Sidebar {
    background-color: #f2f3f7;
    border-right: 1px solid #e8eaef;
}
#SidebarSection { background: transparent; }
#SectionHeader { background: transparent; }
#SectionTitle {
    color: #8b8fa0;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: transparent;
}

/* ─── Buttons ────────────────────────────────────────── */
QPushButton {
    background-color: #e8eaef;
    color: #4a4d5a;
    border: 1px solid #dcdfe6;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #dcdfe6;
    border-color: #6366f1;
    color: #1a1c24;
}
QPushButton:pressed { background-color: #d0d3db; }
QPushButton:disabled { background-color: #f2f3f7; color: #b0b3c0; border-color: #e8eaef; }
QPushButton#AccentBtn { background-color: #6366f1; color: #ffffff; border: none; font-weight: 600; }
QPushButton#AccentBtn:hover { background-color: #818cf8; }
QPushButton#GreenBtn { background-color: #16a34a; color: #ffffff; border: none; font-weight: 600; }
QPushButton#GreenBtn:hover { background-color: #22c55e; }
QPushButton#RedBtn { background-color: #dc2626; color: #ffffff; border: none; font-weight: 600; }
QPushButton#RedBtn:hover { background-color: #ef4444; }
QPushButton#IconBtn { background: transparent; border: none; font-size: 15px; padding: 4px 8px; border-radius: 6px; color: #8b8fa0; }
QPushButton#IconBtn:hover { background-color: #e8eaef; color: #1a1c24; }
QPushButton#ToolbarBtn { background: transparent; border: 1px solid #dcdfe6; border-radius: 8px; padding: 5px 12px; color: #6b6f80; font-size: 12px; }
QPushButton#ToolbarBtn:hover { background-color: #e8eaef; border-color: #6366f1; color: #1a1c24; }
QPushButton#GhostBtn { background: transparent; border: 1px solid #dcdfe6; border-radius: 8px; padding: 5px 14px; color: #6b6f80; font-size: 12px; }
QPushButton#GhostBtn:hover { background-color: #e8eaef; color: #1a1c24; }
QPushButton#TinyBtn { background: transparent; border: none; border-radius: 4px; padding: 2px 4px; font-size: 13px; color: #8b8fa0; }
QPushButton#TinyBtn:hover { background-color: #e8eaef; color: #1a1c24; }
QPushButton#InputIcon { background-color: #f2f3f7; border: 1px solid #dcdfe6; border-radius: 17px; color: #4f5568; font-size: 16px; font-weight: 700; }
QPushButton#InputIcon:hover { background-color: #e8eaef; border-color: #cfd5e2; color: #1a1c24; }
QPushButton#SendBtn { background-color: #6366f1; color: #ffffff; border: none; border-radius: 19px; font-size: 15px; font-weight: bold; }
QPushButton#SendBtn:hover { background-color: #818cf8; }
QPushButton#SendBtn:disabled { background-color: #e8eaef; color: #b0b3c0; }
QPushButton#StopBtn { background-color: #dc2626; color: #ffffff; border: none; border-radius: 17px; font-size: 12px; }
QPushButton#StopBtn:hover { background-color: #ef4444; }
QPushButton#StopBtn:disabled { background-color: transparent; color: #b0b3c0; border: 1px solid #dcdfe6; }
QPushButton#ActionBtn { background: transparent; border: none; color: #8b8fa0; font-size: 11px; padding: 3px 8px; border-radius: 4px; }
QPushButton#ActionBtn:hover { color: #1a1c24; background-color: #e8eaef; }

/* ─── ComboBox ───────────────────────────────────────── */
QComboBox { background-color: #ffffff; color: #1a1c24; border: 1px solid #dcdfe6; border-radius: 8px; padding: 5px 12px; min-height: 26px; }
QComboBox:hover { border-color: #6366f1; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #6366f1; margin-right: 8px; }
QComboBox QAbstractItemView { background-color: #ffffff; color: #1a1c24; border: 1px solid #dcdfe6; selection-background-color: #eef0f4; selection-color: #6366f1; border-radius: 8px; padding: 4px; }

/* ─── Line Edit ──────────────────────────────────────── */
QLineEdit { background-color: #ffffff; color: #1a1c24; border: 1px solid #dcdfe6; border-radius: 8px; padding: 6px 12px; selection-background-color: #6366f1; selection-color: #ffffff; }
QLineEdit:focus { border-color: #6366f1; }
QLineEdit::placeholder { color: #b0b3c0; }
#ChatInput { background-color: #f2f3f7; border: 1px solid #dcdfe6; border-radius: 20px; padding: 8px 16px; font-size: 13px; }
#ChatInput:focus { border-color: #6366f1; }

/* ─── Text Edit ──────────────────────────────────────── */
QTextEdit, QPlainTextEdit { background-color: #ffffff; color: #1a1c24; border: 1px solid #dcdfe6; border-radius: 8px; padding: 8px; selection-background-color: #6366f1; selection-color: #ffffff; }
QTextEdit:focus, QPlainTextEdit:focus { border-color: #6366f1; }

/* ─── Scroll Area ────────────────────────────────────── */
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }

/* ─── Scroll Bar ─────────────────────────────────────── */
QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }
QScrollBar::handle:vertical { background: #d0d3db; min-height: 40px; border-radius: 3px; }
QScrollBar::handle:vertical:hover { background: #b0b3c0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 6px; margin: 0; }
QScrollBar::handle:horizontal { background: #d0d3db; min-width: 40px; border-radius: 3px; }
QScrollBar::handle:horizontal:hover { background: #b0b3c0; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ─── Progress Bar ───────────────────────────────────── */
QProgressBar { background-color: #e8eaef; border: none; border-radius: 2px; max-height: 3px; text-align: center; font-size: 0px; }
QProgressBar::chunk { background-color: #6366f1; border-radius: 2px; }

/* ─── Slider ─────────────────────────────────────────── */
QSlider::groove:horizontal { background: #dcdfe6; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #6366f1; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #818cf8; }
QSlider::sub-page:horizontal { background: #6366f1; border-radius: 2px; }

/* ─── Radio Button ───────────────────────────────────── */
QRadioButton { color: #1a1c24; spacing: 8px; }
QRadioButton::indicator { width: 16px; height: 16px; border-radius: 8px; border: 2px solid #b0b3c0; background-color: #ffffff; }
QRadioButton::indicator:checked { border-color: #6366f1; background-color: #6366f1; }

/* ─── Labels ─────────────────────────────────────────── */
QLabel { color: #1a1c24; background: transparent; }
QLabel#Muted { color: #6f7487; font-size: 11px; }
QLabel#Accent { color: #6366f1; }
QLabel#Gold { color: #d97706; }
QLabel#StatusLabel { color: #4f5568; font-size: 11px; }

/* ─── Chat Area ──────────────────────────────────────── */
#ChatArea { background-color: #fafbfd; border: none; }
#ChatScroll { background-color: #fafbfd; }

/* ─── Chat Bubbles ───────────────────────────────────── */
#UserBubble { background-color: #6366f1; border-radius: 16px; border: none; }
#UserBubble QTextEdit { background: transparent; color: #ffffff; border: none; font-size: 13px; }
#AssistantBubble { background-color: #ffffff; border-radius: 16px; border: 1px solid #d6dae4; }
#AssistantBubble QTextEdit { background: transparent; color: #1a1c24; border: none; font-size: 13px; }

/* ─── Input Bar ──────────────────────────────────────── */
#InputBar { background-color: #fafbfd; border-top: 1px solid #e8eaef; }

/* ─── Chat List ──────────────────────────────────────── */
#ChatRow { background-color: transparent; border-radius: 8px; border: none; }
#ChatRow:hover { background-color: #e3e7f0; }
#ChatRowActive { background-color: #e3e7f0; border-radius: 8px; border: 1px solid #cfd5e2; }

/* ─── RAG Rows ───────────────────────────────────────── */
#RagRow { background-color: transparent; border-radius: 6px; border: none; }
#RagRow:hover { background-color: #e3e7f0; }

/* ─── Dialog ─────────────────────────────────────────── */
QDialog { background-color: #fafbfd; }

/* ─── Splitter ───────────────────────────────────────── */
QSplitter::handle { background-color: #e8eaef; width: 1px; }

/* ─── ToolTip ────────────────────────────────────────── */
QToolTip { background-color: #ffffff; color: #1a1c24; border: 1px solid #dcdfe6; border-radius: 6px; padding: 4px 8px; }

/* ─── Menu ───────────────────────────────────────────── */
QMenu { background-color: #ffffff; color: #1a1c24; border: 1px solid #dcdfe6; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 6px 20px; border-radius: 4px; }
QMenu::item:selected { background-color: #eef0f4; color: #6366f1; }
QMenu::separator { height: 1px; background: #e8eaef; margin: 4px 8px; }
"""


def get_theme(name: str) -> str:
    return DARK_THEME if name == "Dark" else LIGHT_THEME


# ── Color palette helpers ─────────────────────────────────────────────────
COLORS = {
    "Dark": {
        "bg": "#0d1017",
        "surface": "#1a1d28",
        "overlay": "#262a3a",
        "text": "#e2e4ec",
        "subtext": "#8b90a5",
        "muted": "#555a70",
        "accent": "#818cf8",
        "green": "#4ade80",
        "red": "#f87171",
        "yellow": "#fbbf24",
        "peach": "#fb923c",
        "user_bubble": "#6366f1",
        "user_text": "#ffffff",
        "assistant_bubble": "#151820",
        "assistant_text": "#e2e4ec",
        "sidebar": "#111318",
        "input_bg": "#151820",
        "border": "#1c1f2e",
        "code_bg": "#0a0c12",
        "code_fg": "#e2e4ec",
    },
    "Light": {
        "bg": "#fafbfd",
        "surface": "#ffffff",
        "overlay": "#e8eaef",
        "text": "#1a1c24",
        "subtext": "#6b6f80",
        "muted": "#8b8fa0",
        "accent": "#6366f1",
        "green": "#16a34a",
        "red": "#dc2626",
        "yellow": "#d97706",
        "peach": "#ea580c",
        "user_bubble": "#6366f1",
        "user_text": "#ffffff",
        "assistant_bubble": "#f0f1f5",
        "assistant_text": "#1a1c24",
        "sidebar": "#f2f3f7",
        "input_bg": "#f2f3f7",
        "border": "#dcdfe6",
        "code_bg": "#f5f6f8",
        "code_fg": "#1a1c24",
    },
}
