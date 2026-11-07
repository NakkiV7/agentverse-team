"""Is Gemma fast enough and does it give the shapes we need?  uv run python scripts/llm_smoke.py"""
from buddy.agent.llm import LLM

llm = LLM()
print("model:", llm.model, "warming up (first load takes a few seconds)...")
llm.warm()
for s in ["Buddy, let's play a game", "I'll take the middle one", "say that again please",
          "top right corner for me", "that's enough for today thank you", "how are you today?"]:
    it, t = llm.intent(s)
    print(f"  {t['total_s']:.2f}s  {s!r:40} -> {it}")
line, t = llm.speak("The player just drew an O in the centre. Buddy will draw an X in the top left. Say what Buddy says.")
print(f"  {t['total_s']:.2f}s  speak -> {line!r}")
