from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QScrollArea, QVBoxLayout, QWidget

USER_COLORS = ("#2f6fd4", "#ffffff")        # background, text
ASSISTANT_COLORS = ("#2a2a2a", "#e0e0e0")


class MessageRow(QWidget):
    """One chat bubble, aligned right for the user and left for the assistant."""

    def __init__(self, text: str, is_user: bool):
        super().__init__()
        background, foreground = USER_COLORS if is_user else ASSISTANT_COLORS

        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(400)
        bubble.setContentsMargins(12, 8, 12, 8)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bubble.setStyleSheet(
            f"QLabel {{ background-color: {background}; color: {foreground}; border-radius: 14px; font-size: 14px; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        if is_user:
            layout.addStretch()
        layout.addWidget(bubble)
        if not is_user:
            layout.addStretch()


class ChatWindow(QWidget):
    """Frameless, always-on-top chat window."""

    user_message_submitted = pyqtSignal(str)
    hidden = pyqtSignal()

    def __init__(self, ui_config: dict):
        super().__init__()
        self._drag_offset = None

        self.setWindowTitle("AIDE")
        self.setFixedSize(ui_config["window_width"], ui_config["window_height"])
        self.setWindowOpacity(ui_config["opacity"])
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._build_layout()

    def _build_layout(self) -> None:
        frame = QFrame()
        frame.setObjectName("main_frame")
        frame.setStyleSheet(
            "QFrame#main_frame { background-color: #1a1a1a; border-radius: 16px; border: 1px solid #333333; }"
        )
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(0, 12, 0, 12)
        inner.setSpacing(4)

        title = QLabel("AIDE")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #666666; font-size: 11px; font-weight: 600; letter-spacing: 4px;")
        inner.addWidget(title)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("background: transparent;")
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        messages = QWidget()
        messages.setStyleSheet("background: transparent;")
        self._messages = QVBoxLayout(messages)
        self._messages.setSpacing(4)
        self._messages.addStretch()
        self._scroll.setWidget(messages)
        inner.addWidget(self._scroll)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask AIDE something...")
        self.input.setStyleSheet(
            "QLineEdit { background-color: #242424; color: #e0e0e0; border: 1px solid #3a3a3a;"
            " border-radius: 10px; padding: 10px 14px; font-size: 14px; margin: 0 12px; }"
            "QLineEdit:focus { border: 1px solid #2f6fd4; }"
        )
        self.input.returnPressed.connect(self._submit)
        inner.addWidget(self.input)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

    # ---- public API ----

    def add_message(self, text: str, is_user: bool) -> None:
        self._messages.insertWidget(self._messages.count() - 1, MessageRow(text, is_user))
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def set_thinking(self, thinking: bool) -> None:
        self.input.setEnabled(not thinking)
        self.input.setPlaceholderText("AIDE is thinking…" if thinking else "Ask AIDE something...")
        if not thinking:
            self.input.setFocus()

    def toggle(self) -> None:
        if self.isVisible():
            self.hide()
            return
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 3)
        self.show()
        self.raise_()
        self.input.setFocus()

    # ---- events ----

    def _submit(self) -> None:
        text = self.input.text().strip()
        if text:
            self.input.clear()
            self.add_message(text, is_user=True)
            self.user_message_submitted.emit(text)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.hidden.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
