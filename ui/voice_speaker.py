"""Text-to-speech with Kokoro, played through aplay/paplay (optional extra)."""

import logging
import os
import re
import subprocess
import tempfile
import threading
import wave

import numpy as np
from kokoro import KPipeline

log = logging.getLogger("aide.speaker")

SAMPLE_RATE = 24_000


def normalize_for_tts(text: str) -> str:
    """Spell out symbols that TTS engines read badly: file names, paths, dashes."""
    text = re.sub(r"(\w+)\.(\w+)", r"\1 dot \2", text)
    text = text.replace("_", " underscore ").replace("/", " slash ").replace("\\", " backslash ")
    text = text.replace("—", "…").replace("–", "…").replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"([.!?…])(\S)", r"\1 \2", text)   # "end.Next" -> "end. Next"
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]


class VoiceSpeaker:
    def __init__(self, voice: str = "af_sky", pause: float = 0.2):
        self.voice = voice
        self.pause = pause
        self._pipeline = KPipeline(lang_code="a")
        self._lock = threading.Lock()   # one utterance at a time

    def say(self, text: str, block: bool = False) -> None:
        """Speak `text`. With block=False it returns immediately; block=True waits until playback ends."""
        thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
        thread.start()
        if block:
            thread.join()

    def _speak(self, text: str) -> None:
        with self._lock:
            try:
                silence = np.zeros(int(self.pause * SAMPLE_RATE))
                pieces = []
                for sentence in split_sentences(normalize_for_tts(text)):
                    for _, _, audio in self._pipeline(sentence, voice=self.voice, speed=1.0):
                        pieces += [audio, silence]
                if pieces:
                    self._play(np.concatenate(pieces))
            except Exception:   # TTS and audio playback must never crash the app
                log.exception("speech failed")

    @staticmethod
    def _play(audio: np.ndarray) -> None:
        pcm = (audio * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            with wave.open(path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(pcm.tobytes())
            try:
                subprocess.run(["aplay", path], check=True, capture_output=True)
            except FileNotFoundError:
                subprocess.run(["paplay", path], check=True, capture_output=True)
        finally:
            os.unlink(path)
