"""Owner: Voice and stage.
Microphone -> transcripts (Silero VAD + whisper.cpp); Say -> speech (Kokoro-82M).

Hour 11 milestone:
  out: Kokoro-82M (pip install kokoro soundfile sounddevice), sentence by sentence,
       slightly slower rate (speed=0.85), British voice (bf_emma); "urgent" lines interrupt.
  in:  Silero VAD + whisper.cpp small.en (brew install whisper-cpp), plus a push-to-talk key.
       Wake word "Buddy" via case-insensitive substring check.
       Graceful fallback to console print & typed input if dependencies are missing.
"""
from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from buddy.bus import Bus
from buddy.contracts import Intent, Say

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_is_buddy_speaking = False


class Speaker:
    """Kokoro-82M text-to-speech engine.

    Synthesizes sentence by sentence at a slightly slower rate with a British voice.
    Urgent lines interrupt any currently playing audio immediately.
    """

    SAMPLE_RATE = 24000
    VOICE = "bf_emma"       # British female voice
    LANG_CODE = "b"         # British English
    SPEED = 0.85            # Slightly slower, calm delivery for older players

    def __init__(self, bus: Bus):
        self.bus = bus
        self.pipe = None
        self._kokoro_ready = False
        self._sd = None
        self._interrupt_event = threading.Event()
        self._current_priority = "normal"
        self._lock = threading.Lock()

        self._init_tts()

    def _init_tts(self) -> None:
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            self._sd = None

        try:
            from kokoro import KPipeline
            self.pipe = KPipeline(lang_code=self.LANG_CODE)
            self._kokoro_ready = True
            print("[voice/speaker] Kokoro-82M TTS loaded (British voice bf_emma, speed 0.85).", flush=True)
        except Exception as e:
            self.pipe = None
            self._kokoro_ready = False
            print(f"[voice/speaker] Kokoro TTS not available ({e}). Using console fallback.", flush=True)

    def _split_sentences(self, text: str) -> list[str]:
        sents = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
        return sents if sents else [text.strip()]

    def _play_sentence(self, sentence: str) -> bool:
        global _is_buddy_speaking
        if self._interrupt_event.is_set():
            return False

        if not self._kokoro_ready or self._sd is None:
            _is_buddy_speaking = True
            try:
                words = len(sentence.split())
                dur = max(0.5, words * 0.35)
                steps = int(dur / 0.05)
                for _ in range(steps):
                    if self._interrupt_event.is_set():
                        return False
                    time.sleep(0.05)
            finally:
                _is_buddy_speaking = False
            return True

        try:
            chunks = []
            for _, _, audio in self.pipe(
                sentence,
                voice=self.VOICE,
                speed=self.SPEED,
                split_pattern=None,
            ):
                if hasattr(audio, "numpy"):
                    chunks.append(audio.numpy())
                else:
                    chunks.append(np.asarray(audio, dtype=np.float32))

            if not chunks:
                return True

            audio_data = np.concatenate(chunks)
            if self._interrupt_event.is_set():
                return False

            _is_buddy_speaking = True
            try:
                self._sd.play(audio_data, self.SAMPLE_RATE)
                total_dur = len(audio_data) / self.SAMPLE_RATE
                start_t = time.monotonic()
                while time.monotonic() - start_t < total_dur:
                    if self._interrupt_event.is_set():
                        self._sd.stop()
                        return False
                    time.sleep(0.03)
            finally:
                _is_buddy_speaking = False

            return True

        except Exception as e:
            print(f"[voice/speaker] Audio playback error: {e}", flush=True)
            _is_buddy_speaking = False
            return True

    def run(self) -> None:
        while not self.bus.stop.is_set():
            say_item = self.bus.say.get(timeout=0.1)
            if say_item is None:
                continue

            text = say_item.text.strip()
            priority = say_item.priority
            if not text:
                continue

            if priority == "urgent":
                self._interrupt_event.set()
                if self._sd is not None:
                    try:
                        self._sd.stop()
                    except Exception:
                        pass
                self._interrupt_event.clear()

            print(f"[say/{priority}] {text}", flush=True)

            sentences = self._split_sentences(text)
            for s in sentences:
                if self._interrupt_event.is_set() and priority != "urgent":
                    break
                ok = self._play_sentence(s)
                if not ok:
                    break


class Listener:
    """Silero VAD + whisper.cpp speech recognition listener.

    Listens for speech, ignores self-echo while Buddy speaks, checks for wake word 'Buddy',
    and converts utterances to Intents via LLM or rule parsing.
    """

    SAMPLE_RATE = 16000
    FRAME_SIZE = 512

    def __init__(self, bus: Bus, llm=None):
        self.bus = bus
        self.llm = llm
        self.enabled = True
        self.model = None
        self.vad_iterator = None
        self._whisper_bin = self._find_whisper_cpp()
        self._last_speech_time = 0.0

        self._init_vad()

    def _find_whisper_cpp(self) -> Optional[str]:
        candidates = [
            "whisper-cpp",
            "whisper-cli",
            "whisper",
            "main",
            "/opt/homebrew/bin/whisper-cpp",
            "/usr/local/bin/whisper-cpp",
            "C:\\Program Files\\whisper-cpp\\whisper-cpp.exe",
        ]
        for c in candidates:
            p = shutil.which(c)
            if p:
                return p
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        return None

    def _init_vad(self) -> None:
        try:
            from silero_vad import VADIterator, load_silero_vad
            self.model = load_silero_vad()
            self.vad_iterator = VADIterator(
                self.model,
                sampling_rate=self.SAMPLE_RATE,
                threshold=0.5,
                min_silence_duration_ms=600,
                speech_pad_ms=200,
            )
            print("[voice/listener] Silero VAD initialized.", flush=True)
        except Exception:
            try:
                import torch
                model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
                (_, _, _, VADIteratorHub, _) = utils
                self.model = model
                self.vad_iterator = VADIteratorHub(model, sampling_rate=self.SAMPLE_RATE)
                print("[voice/listener] Silero VAD loaded via torch.hub.", flush=True)
            except Exception as e:
                self.model = None
                self.vad_iterator = None
                print(f"[voice/listener] VAD not available ({e}).", flush=True)

    def _transcribe(self, audio_data: np.ndarray) -> tuple[str, float]:
        if not self._whisper_bin:
            return "", 0.0

        t0 = time.time()
        try:
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                wav_path = tf.name
                sf.write(wav_path, audio_data, self.SAMPLE_RATE, subtype="PCM_16")

            model_candidates = [
                Path("models/ggml-small.en.bin"),
                Path.home() / ".cache/whisper/ggml-small.en.bin",
                Path("/opt/homebrew/share/whisper-cpp/models/ggml-small.en.bin"),
            ]
            model_path = None
            for mc in model_candidates:
                if mc.is_file():
                    model_path = mc
                    break

            cmd = [self._whisper_bin]
            if model_path:
                cmd.extend(["-m", str(model_path)])
            cmd.extend(["-f", wav_path, "-nt", "-np", "-l", "en", "-t", "4"])

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            Path(wav_path).unlink(missing_ok=True)

            text = " ".join(line.strip() for line in proc.stdout.splitlines() if line.strip())
            text = text.replace("[BLANK_AUDIO]", "").strip()
            return text, time.time() - t0
        except Exception as e:
            print(f"[voice/listener] Whisper transcription error: {e}", flush=True)
            return "", 0.0

    def _process_utterance(self, text: str) -> None:
        text_clean = text.strip()
        if not text_clean:
            return

        t_lower = text_clean.lower()
        now = time.monotonic()

        has_wake_word = "buddy" in t_lower
        is_recent_interaction = (now - self._last_speech_time) < 25.0

        if not (has_wake_word or is_recent_interaction):
            return

        self._last_speech_time = now

        cleaned = re.sub(r"(?i)\bbuddy[,!?]?\s*", "", text_clean).strip()
        if not cleaned:
            cleaned = "hello"

        print(f"[voice/listener] Heard: \"{text_clean}\" -> processing: \"{cleaned}\"", flush=True)
        self.bus.transcript.put(f"Player: {text_clean}")

        intent_obj = None
        if self.llm is not None:
            try:
                intent_obj, timing = self.llm.intent(cleaned)
            except Exception as e:
                print(f"[voice/listener] LLM error: {e}", flush=True)

        if intent_obj is None:
            c_low = cleaned.lower()
            if "stop" in c_low or "enough" in c_low:
                intent_obj = Intent(kind="stop", text=cleaned)
            elif "play" in c_low or "yes" in c_low or "game" in c_low:
                intent_obj = Intent(kind="play", text=cleaned)
            elif "repeat" in c_low or "again" in c_low:
                intent_obj = Intent(kind="repeat", text=cleaned)
            else:
                m = re.search(r"\b([0-8])\b", cleaned)
                if m:
                    intent_obj = Intent(kind="move", cell=int(m.group(1)), text=cleaned)
                else:
                    intent_obj = Intent(kind="chat", text=cleaned)

        self.bus.intents.put(intent_obj)

    def run(self) -> None:
        try:
            import sounddevice as sd
            import torch
        except ImportError:
            self.bus.stop.wait()
            return

        if self.model is None or self.vad_iterator is None:
            self.bus.stop.wait()
            return

        audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass
            audio_q.put(indata[:, 0].copy())

        try:
            stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                blocksize=self.FRAME_SIZE,
                dtype="float32",
                callback=audio_callback,
            )
            with stream:
                speech_buffer: list[np.ndarray] = []
                is_active = False

                while not self.bus.stop.is_set():
                    try:
                        frame = audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if _is_buddy_speaking or not self.enabled:
                        speech_buffer.clear()
                        is_active = False
                        self.vad_iterator.reset_states()
                        continue

                    vad_out = self.vad_iterator(torch.from_numpy(frame), return_seconds=False)
                    if vad_out and "start" in vad_out:
                        is_active = True
                        speech_buffer = [frame]

                    elif is_active:
                        speech_buffer.append(frame)

                        if vad_out and "end" in vad_out:
                            is_active = False
                            audio_seg = np.concatenate(speech_buffer)
                            speech_buffer = []

                            if len(audio_seg) >= int(0.3 * self.SAMPLE_RATE):
                                text, _ = self._transcribe(audio_seg)
                                if text:
                                    self._process_utterance(text)

        except Exception as e:
            print(f"[voice/listener] Audio input error: {e}", flush=True)
            self.bus.stop.wait()
