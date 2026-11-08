"""Check the voice stack and print its numbers.

    .venv/bin/python tools/voice_check.py                    # every engine, playback on, mic if there is one
    .venv/bin/python tools/voice_check.py --engines piper    # one engine
    .venv/bin/python tools/voice_check.py --quiet --no-mic   # nothing played, the clip comes from the TTS

Speech: three sentences through each engine (piper, kokoro, say): time to first audio, total synthesis time, audio
length, then played (unless --quiet). Hearing: a 5-second clip is recorded from the microphone (speak when asked), or
synthesised with the first engine when there is no microphone or --no-mic, and fed through the Listener at real-time
pace: the partial transcriptions are printed as they arrive, then the final text and its time.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SENTENCES = ("Lovely, let's play hangman then.",
             "I'll think of a word with six letters, and you can guess it one letter at a time.",
             "Tell me a letter whenever you're ready.")
CLIP_TEXT = "Buddy, let's play a game of hangman please, and make it an easy one."


def check_speech(engines: list[str], quiet: bool):
    from buddy.voice.tts import Speaker
    first = None
    for name in engines:
        events = []
        t0 = time.time()
        try:
            sp = Speaker(on_event=lambda k, t: events.append(t), play=not quiet, engine=name)
        except Exception as e:  # noqa: BLE001
            print(f"{name:7} could not load: {e}")
            continue
        if sp.engine.name != name:
            print(f"{name:7} not available ({'; '.join(e for e in events if 'unavailable' in e)}), skipped")
            continue
        print(f"{name:7} {sp.label}: loaded and warm in {sp.load_s:.2f} s")
        firsts, totals, audio_s = [], [], 0.0
        for s in SENTENCES:
            t1 = time.time()
            a = sp.engine.synth(s)
            totals.append(time.time() - t1)
            audio_s += len(a) / sp.sr
        # first-audio as the Speaker reports it: the whole three-sentence reply queued at once, first piece ready
        n = len(events)
        done = sp.say(" ".join(SENTENCES))
        done.wait(60)
        ready = [e for e in events[n:] if "voice ready" in e]
        first_ms = sp.first_audio_ms[-1] if sp.first_audio_ms else float("nan")
        print(f"        first audio {first_ms:.0f} ms, synthesis of the three sentences "
              f"{', '.join(f'{1000 * t:.0f}' for t in totals)} ms (total {1000 * sum(totals):.0f} ms), "
              f"audio {audio_s:.1f} s, played in {time.time() - t0 - sp.load_s:.1f} s")
        for e in events:
            if "voice ready" not in e:
                print(f"        note: {e}")
        if first is None:
            first = sp
    return first


def record_clip(seconds: float):
    import sounddevice as sd
    from buddy.voice.stt import SR
    try:
        sd.check_input_settings(samplerate=SR, channels=1)
    except Exception as e:  # noqa: BLE001
        print(f"no microphone ({e})")
        return None
    print(f"recording {seconds:.0f} s from the microphone: say something like '{CLIP_TEXT}' ...")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0].copy()


def synth_clip(speaker, seconds: float):
    from buddy.voice.stt import SR
    if speaker is None:
        from buddy.voice.tts import Speaker
        speaker = Speaker(play=False)
    a = speaker.engine.synth(CLIP_TEXT)
    # resample to 16 kHz by linear interpolation (good enough for a check)
    n = int(len(a) * SR / speaker.sr)
    a16 = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    pad = np.zeros(int(0.5 * SR), np.float32)
    clip = np.concatenate([pad, a16, pad])
    return clip[: int(seconds * SR)] if len(clip) > seconds * SR else clip


def check_hearing(clip: np.ndarray):
    from buddy.voice.stt import FRAME, SR, Listener
    t0 = time.time()
    start = {"t": 0.0}
    out = []

    def on_text(text, dt):
        out.append(text)
        print(f"        final   at {time.time() - start['t']:.2f} s ({1000 * dt:.0f} ms): {text!r}")

    def on_partial(text, dt):
        print(f"        partial at {time.time() - start['t']:.2f} s ({1000 * dt:.0f} ms): {text!r}")

    lst = Listener(on_text=on_text, is_speaking=lambda: False, on_partial=on_partial)
    print(f"hearing {lst.label}: loaded in {time.time() - t0:.1f} s (final model {lst.final.load_s:.1f} s"
          + (f", partial model {lst.partial_model.load_s:.1f} s)" if lst.partial_model else ")"))
    lst.start_threads()
    clip = np.concatenate([clip, np.zeros(SR, np.float32)])       # a second of silence ends the utterance
    start["t"] = time.time()
    for i in range(0, len(clip) - FRAME, FRAME):
        lst.feed(clip[i:i + FRAME])
        time.sleep(FRAME / SR)
    t_end = time.time() + 5
    while time.time() < t_end and (lst.hearing()["active"] or not out):
        time.sleep(0.05)
    print(f"        hearing field at rest: {lst.hearing()}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="piper,kokoro,say", help="comma-separated: piper, kokoro, say")
    ap.add_argument("--quiet", action="store_true", help="do not play the sentences")
    ap.add_argument("--no-mic", action="store_true", help="synthesise the test clip instead of recording it")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    speaker = check_speech(engines, args.quiet)
    clip = None if args.no_mic else record_clip(args.seconds)
    if clip is None or float(np.sqrt(np.mean(clip ** 2))) < 1e-4:
        if clip is not None:
            print("the recording is silent: using a synthesised clip instead")
        clip = synth_clip(speaker, args.seconds)
        print(f"clip: synthesised, {len(clip) / 16000:.1f} s, '{CLIP_TEXT}'")
    else:
        print(f"clip: recorded, {len(clip) / 16000:.1f} s")
    out = check_hearing(clip)
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
