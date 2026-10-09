"""Listens on a Unix socket so desktop keyboard shortcuts (scripts/*.sh) can control AIDE."""

import logging
import os
import socket

from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger("aide.hotkeys")

SOCKET_PATH = "/tmp/aide.sock"


class HotkeyDaemon(QObject):
    toggle_window = pyqtSignal()
    toggle_recording = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = False

    def start(self) -> None:
        """Blocking accept loop; run it on a worker QThread."""
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(1)
        server.settimeout(1.0)   # wake up regularly so stop() takes effect
        self._running = True
        log.info("listening on %s", SOCKET_PATH)

        while self._running:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                command = conn.recv(32).decode().strip()
            if command == "toggle":
                self.toggle_window.emit()
            elif command == "record_toggle":
                self.toggle_recording.emit()

        server.close()
        os.remove(SOCKET_PATH)

    def stop(self) -> None:
        self._running = False
