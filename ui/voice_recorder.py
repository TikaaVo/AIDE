"""Push-to-toggle microphone capture with local Whisper transcription (optional extra)."""

import logging
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

log = logging.getLogger("aide.recorder")

SAMPLE_RATE = 16_000


class VoiceRecorder:
    def __init__(self, on_transcription: Callable[[str], None], model: str, vocabulary: str = ""):
        self.on_transcription = on_transcription
        self.vocabulary = vocabulary
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        log.info("loading Whisper model %s...", model)
        self._whisper = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=8)

    def toggle_recording(self) -> None:
        if self._stream is None:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
            callback=lambda indata, *_: self._frames.append(indata.copy()),
        )
        self._stream.start()
        log.info("recording...")

    def _stop(self) -> None:
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if self._frames:
            audio = np.concatenate(self._frames).flatten()
            threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: np.ndarray) -> None:
        segments, _ = self._whisper.transcribe(
            audio,
            language="en",
            initial_prompt=self.vocabulary or None,
            beam_size=3,
            vad_filter=True,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text for seg in segments).strip()
        if text:
            log.info("heard: %r", text)
            self.on_transcription(text)
