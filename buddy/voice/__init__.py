"""Owner: Voice and stage. Microphone -> transcripts (Silero VAD + whisper.cpp); Say -> speech (Kokoro-82M)."""
from buddy.voice.voice import Listener, Speaker

__all__ = ["Speaker", "Listener"]
