"""The voice stack without a microphone or a loudspeaker: text splitting, the speaker's threads and busy() contract
with a fake engine and a fake output stream (including a PortAudio failure and the reopen), the listener's partial and
final transcriptions from fed audio with fake whisper models, the hearing field, and echo suppression."""
import sys
import threading
import time
import types

import numpy as np
import pytest

from buddy import config as C
from buddy.voice import stt, tts
from buddy.voice.stt import FRAME, SR, Listener, clean
from buddy.voice.tts import Speaker, fade, split_text


# ------------------------------------------------------------------------------------------------- text
def test_split_sentences_unchanged_for_fast_engines():
    text = "Lovely, let's play hangman then. I'll think of a word. Tell me a letter whenever you're ready."
    assert split_text(text) == ["Lovely, let's play hangman then.", "I'll think of a word.",
                                "Tell me a letter whenever you're ready."]
    assert split_text(text, split_first=True) == split_text(text)      # the first sentence is short: no clause split


def test_split_long_first_sentence_into_clauses_for_slow_engines():
    text = ("I know noughts and crosses, Footsteps, backwards noughts and crosses, Footsteps with diagonals and "
            "hangman, and you can teach me a new one. Which would you like?")
    pieces = split_text(text, split_first=True)
    assert pieces[-1] == "Which would you like?"
    assert len(pieces) >= 4 and all(len(p.split()) >= tts.MIN_PIECE for p in pieces[:-1])
    assert " ".join(pieces) == text                                     # nothing lost, nothing reordered


def test_split_empty_and_whitespace():
    assert split_text("") == [] and split_text("   ") == []


def test_fade_keeps_length_and_softens_edges():
    a = np.ones(2400, np.float32)
    f = fade(a, 24000)
    assert len(f) == 2400 and f[0] < 0.01 and f[-1] < 0.01 and f[1200] == 1.0
    assert len(fade(np.ones(3, np.float32), 24000)) == 3                # too short to fade: unchanged length


def test_clean_drops_noise_tokens():
    assert clean(" [BLANK_AUDIO] Buddy, (typing) let's play. *music* ") == "Buddy, let's play."
    assert clean("[BLANK_AUDIO]") == ""


# ------------------------------------------------------------------------------------------------- speaker
class FakeEngine:
    name, slow, sr, label = "fake", False, 16000, "fake engine"

    def __init__(self, delay=0.0):
        self.delay, self.calls = delay, []

    def synth(self, text):
        self.calls.append(text)
        time.sleep(self.delay)
        return np.full(int(0.05 * self.sr), 0.1, np.float32)          # 50 ms of audio a piece

    def warm(self):
        pass


class FakeStream:
    """Stands in for sounddevice.OutputStream; `fail` streams raise on write like a device that went away."""
    opened, written = [], []

    def __init__(self, samplerate, channels, dtype):
        self.latency = 0.0
        self.fail = FakeStream.fail_next
        FakeStream.fail_next = False
        FakeStream.opened.append(samplerate)

    def start(self):
        pass

    def write(self, data):
        if self.fail:
            raise RuntimeError("Error starting stream: Internal PortAudio error [PaErrorCode -9986]")
        FakeStream.written.append(len(data))

    def stop(self):
        pass

    def close(self):
        pass


FakeStream.fail_next = False


@pytest.fixture
def fake_sd(monkeypatch):
    FakeStream.opened, FakeStream.written, FakeStream.fail_next = [], [], False
    sd = types.SimpleNamespace(OutputStream=FakeStream, _terminate=lambda: None, _initialize=lambda: None,
                               resets=[])
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def make_speaker(monkeypatch, engine, play=False):
    monkeypatch.setattr(tts, "make_engine", lambda name=None: (engine, []))
    events = []
    sp = Speaker(on_event=lambda k, t: events.append((k, t)), play=play)
    return sp, events


def test_speaker_says_sentence_by_sentence_and_reports_first_audio(monkeypatch):
    eng = FakeEngine(delay=0.02)
    sp, events = make_speaker(monkeypatch, eng)
    done = sp.say("One. Two. Three.")
    assert sp.busy()
    assert done.wait(5)
    time.sleep(0.05)
    assert eng.calls == ["One.", "Two.", "Three."]
    ready = [t for k, t in events if "voice ready" in t]
    assert len(ready) == 1 and ready[0].endswith("One. Two. Three.")
    assert len(sp.first_audio_ms) == 1 and 10 < sp.first_audio_ms[0] < 500
    assert not sp.busy() and sp.last_audio_end > 0


def test_speaker_busy_covers_synthesis_gap(monkeypatch):
    """Between the last played chunk and the next sentence's audio the speaker still counts as busy, so the
    microphone stays shut while Buddy is mid-reply."""
    eng = FakeEngine(delay=0.15)
    sp, _ = make_speaker(monkeypatch, eng)
    done = sp.say("Short one. Short two.")
    samples = []
    t_end = time.time() + 1.5
    while not done.is_set() and time.time() < t_end:
        samples.append(sp.busy())
        time.sleep(0.01)
    assert all(samples) and done.is_set()


def test_speaker_empty_text_sets_event(monkeypatch):
    sp, _ = make_speaker(monkeypatch, FakeEngine())
    assert sp.say("   ").wait(2)


def test_speaker_plays_through_one_stream_and_reopens_after_a_portaudio_error(monkeypatch, fake_sd):
    eng = FakeEngine()
    sp, events = make_speaker(monkeypatch, eng, play=True)
    assert sp.say("One. Two.").wait(5)
    time.sleep(0.05)
    assert len(FakeStream.written) == 2 and len(FakeStream.opened) == 1      # one stream for the whole reply
    FakeStream.fail_next = True                                              # the device changed: first write fails
    assert sp.say("Three.").wait(5)
    time.sleep(0.05)
    assert len(FakeStream.written) == 3                                      # the chunk was played after reopening
    assert any("reopening the output device" in t for _, t in events)
    assert not any("audio output unavailable" in t for _, t in events)


def test_speaker_keeps_timing_when_audio_is_unavailable(monkeypatch, fake_sd):
    class Dead(FakeStream):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.fail = True
    fake_sd.OutputStream = Dead
    sp, events = make_speaker(monkeypatch, FakeEngine(), play=True)
    t0 = time.time()
    assert sp.say("One.").wait(5)
    assert time.time() - t0 >= 0.05                                          # waited the chunk's length
    assert any("audio output unavailable" in t for _, t in events)


def test_make_engine_falls_back_in_order(monkeypatch):
    class Boom:
        def __init__(self):
            raise ImportError("no such package")
    monkeypatch.setattr(tts, "ENGINES", {"piper": Boom, "kokoro": Boom, "say": FakeEngine})
    eng, errors = tts.make_engine("piper")
    assert isinstance(eng, FakeEngine) and [e.split(":")[0] for e in errors] == ["piper", "kokoro"]
    monkeypatch.setattr(tts, "ENGINES", {"piper": Boom, "kokoro": Boom, "say": Boom})
    with pytest.raises(RuntimeError):
        tts.make_engine("piper")


# ------------------------------------------------------------------------------------------------- listener
class FakeWhisper:
    """Returns the number of seconds of audio it was given, so partials can be checked exactly."""

    def __init__(self, name, delay=0.0):
        self.name, self.delay, self.load_s, self.calls = name, delay, 0.0, []

    def transcribe(self, audio):
        self.calls.append(len(audio) / SR)
        time.sleep(self.delay)
        return f"{self.name} heard {len(audio) / SR:.1f} s"


class FakeVAD:
    """Speech is any frame whose RMS is above 0.05; the real VADIterator's events: {'start': n} and {'end': n}."""

    def __init__(self):
        self.in_speech = False

    def __call__(self, frame, return_seconds=False):
        loud = float(np.sqrt(np.mean(frame.numpy() ** 2))) > 0.05
        if loud and not self.in_speech:
            self.in_speech = True
            return {"start": 0}
        if not loud and self.in_speech:
            self.in_speech = False
            return {"end": 0}
        return None

    def reset_states(self):
        self.in_speech = False


def make_listener(monkeypatch, speaking=lambda: False, partial=True, final_delay=0.0):
    monkeypatch.setitem(sys.modules, "silero_vad", types.SimpleNamespace(
        load_silero_vad=lambda: None, VADIterator=lambda *a, **k: FakeVAD()))
    monkeypatch.setattr(C, "STT_PARTIAL_S", 0.1)
    monkeypatch.setattr(C, "STT_MIN_PARTIAL_S", 0.2)
    heard, partials = [], []
    lst = Listener(on_text=lambda t, dt: heard.append(t), is_speaking=speaking,
                   final=FakeWhisper("small", final_delay), partial=FakeWhisper("base") if partial else None,
                   on_partial=lambda t, dt: partials.append(t))
    lst.start_threads()
    return lst, heard, partials


def feed(lst, seconds, loud, realtime=True):
    rng = np.random.default_rng(0)
    n = int(seconds * SR / FRAME)
    for _ in range(n):
        frame = (0.3 * rng.standard_normal(FRAME) if loud else np.zeros(FRAME)).astype(np.float32)
        lst.feed(frame)
        if realtime:
            time.sleep(FRAME / SR)


def test_listener_partials_then_final_and_hearing_field(monkeypatch):
    lst, heard, partials = make_listener(monkeypatch)
    assert lst.hearing() == {"active": False, "partial": "", "level": 0.0}
    feed(lst, 1.0, loud=True)
    h = lst.hearing()
    assert h["active"] and h["partial"].startswith("base heard") and h["level"] > 0.05
    feed(lst, 0.3, loud=False)
    time.sleep(0.3)
    assert heard == ["small heard 1.0 s"]                                   # the final came from the big model
    assert len(partials) >= 3 and all(p.startswith("base heard") for p in partials)
    assert lst.hearing()["active"] is False and lst.hearing()["partial"] == ""
    assert lst.partial_model.calls == sorted(lst.partial_model.calls)      # each partial saw more audio


def test_listener_short_blip_is_ignored(monkeypatch):
    lst, heard, partials = make_listener(monkeypatch)
    feed(lst, 0.2, loud=True)
    feed(lst, 0.2, loud=False)
    time.sleep(0.2)
    assert heard == [] and lst.hearing()["active"] is False


def test_listener_drops_audio_while_buddy_speaks(monkeypatch):
    speaking = {"on": True}
    lst, heard, partials = make_listener(monkeypatch, speaking=lambda: speaking["on"])
    feed(lst, 0.8, loud=True)
    assert lst.hearing()["active"] is False and partials == []
    feed(lst, 0.2, loud=False)
    time.sleep(0.2)
    assert heard == []
    speaking["on"] = False                                                  # Buddy went quiet: the player is heard
    feed(lst, 0.6, loud=True)
    feed(lst, 0.2, loud=False)
    time.sleep(0.2)
    assert heard == ["small heard 0.6 s"]


def test_listener_speaking_mid_utterance_cancels_it(monkeypatch):
    speaking = {"on": False}
    lst, heard, partials = make_listener(monkeypatch, speaking=lambda: speaking["on"])
    feed(lst, 0.5, loud=True)
    assert lst.hearing()["active"]
    speaking["on"] = True
    feed(lst, 0.1, loud=True)
    assert lst.hearing() == {"active": False, "partial": "", "level": round(lst.level, 4)}
    speaking["on"] = False
    feed(lst, 0.2, loud=False)
    time.sleep(0.2)
    assert heard == []


def test_listener_muted_hears_nothing(monkeypatch):
    lst, heard, partials = make_listener(monkeypatch)
    lst.enabled = False
    feed(lst, 0.6, loud=True)
    feed(lst, 0.2, loud=False)
    time.sleep(0.2)
    assert heard == [] and partials == [] and lst.hearing()["active"] is False


def test_listener_without_partial_model_still_transcribes(monkeypatch):
    lst, heard, partials = make_listener(monkeypatch, partial=False)
    assert lst.partial_model is None and "live text" not in lst.label
    feed(lst, 0.6, loud=True)
    feed(lst, 0.2, loud=False)
    time.sleep(0.2)
    assert heard == ["small heard 0.6 s"] and partials == []


def test_listener_active_until_final_delivered(monkeypatch):
    lst, heard, partials = make_listener(monkeypatch, final_delay=0.3)
    feed(lst, 0.6, loud=True)
    feed(lst, 0.2, loud=False, realtime=False)
    time.sleep(0.1)
    assert lst.hearing()["active"] and lst.transcribing                     # the final is still running
    time.sleep(0.5)
    assert heard == ["small heard 0.6 s"] and lst.hearing()["active"] is False


# ------------------------------------------------------------------------------------------------- app contract
def test_app_state_hearing_field_shape():
    """What the page reads: app.state()['hearing'] = {active, partial, level}, with or without a microphone."""
    import inspect
    from buddy import app as A
    src = inspect.getsource(A.App.state)
    assert '"hearing": self.listener.hearing() if self.listener else {"active": False, "partial": "", "level": 0.0}' in src
