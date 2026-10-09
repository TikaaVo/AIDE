"""AIDE entry point: a floating chat window, optional voice, driven by global hotkeys."""

import logging
import sys
import threading

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

from core.config import load_config
from core.llm import LLM
from core.orchestrator import Orchestrator
from tools.registry import build_registry
from ui.chat_window import ChatWindow
from ui.hotkey_daemon import HotkeyDaemon

log = logging.getLogger("aide")


class Bridge(QObject):
    """Carries results from worker threads back to the Qt main thread."""
    message = pyqtSignal(str)       # interim update (acknowledgement)
    finished = pyqtSignal(str)      # final reply
    voice_command = pyqtSignal(str)


def build_voice(cfg: dict, on_transcription):
    """Returns (speaker, recorder), or (None, None) when voice is disabled."""
    if not cfg.get("enabled"):
        return None, None
    from ui.voice_recorder import VoiceRecorder   # heavy imports, only when needed
    from ui.voice_speaker import VoiceSpeaker

    return (
        VoiceSpeaker(cfg["tts_voice"]),
        VoiceRecorder(on_transcription, cfg["whisper_model"], cfg.get("vocabulary", "")),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    window = ChatWindow(cfg["ui"])
    bridge = Bridge()
    speaker, recorder = build_voice(cfg.get("voice", {}), bridge.voice_command.emit)

    llm = LLM.from_config(cfg["llm"])
    tools = build_registry(llm, LLM.from_config(cfg["embedder"]))
    orchestrator = Orchestrator(llm, tools, speaker)
    threading.Thread(target=tools["file_search"].index, daemon=True).start()

    def ask(text: str, voice_mode: bool) -> None:
        """Run one turn off the UI thread."""
        window.set_thinking(True)

        def work() -> None:
            try:
                reply = orchestrator.handle(text, bridge.message.emit, voice_mode)
            except Exception as e:   # surface anything unexpected in the window instead of dying silently
                log.exception("turn failed")
                reply = f"Something went wrong: {e}"
            bridge.finished.emit(reply)

        threading.Thread(target=work, daemon=True).start()

    def on_voice_command(text: str) -> None:
        window.add_message(text, is_user=True)
        ask(text, voice_mode=True)

    bridge.message.connect(lambda text: window.add_message(text, is_user=False))
    bridge.finished.connect(lambda text: (window.add_message(text, is_user=False), window.set_thinking(False)))
    bridge.voice_command.connect(on_voice_command)
    window.user_message_submitted.connect(lambda text: ask(text, voice_mode=False))
    window.hidden.connect(orchestrator.reset)   # closing the window starts a fresh conversation

    hotkeys = HotkeyDaemon()
    hotkey_thread = QThread()
    hotkeys.moveToThread(hotkey_thread)
    hotkeys.toggle_window.connect(window.toggle)
    if recorder:
        hotkeys.toggle_recording.connect(recorder.toggle_recording)
    hotkey_thread.started.connect(hotkeys.start)
    hotkey_thread.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
