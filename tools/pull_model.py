"""Download a model into the local Ollama server, with a live progress bar.

    python tools/pull_model.py gemma3:4b

Needs `ollama serve` (or the Ollama app) running. Progress: runtime/progress.txt and
http://localhost:8850/progress.html (start `python tools/serve_progress.py`).
"""
import json
import sys
import urllib.request

from progress import Progress


def pull(name: str) -> None:
    prog = Progress(f"pull {name}", unit="B")
    req = urllib.request.Request("http://127.0.0.1:11434/api/pull", data=json.dumps({"model": name}).encode(),
                                 headers={"Content-Type": "application/json"})
    layers: dict[str, tuple[int, int]] = {}
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            msg = json.loads(line)
            if "error" in msg:
                raise SystemExit(f"ollama: {msg['error']}")
            if msg.get("digest") and msg.get("total"):
                layers[msg["digest"]] = (msg.get("completed", 0), msg["total"])
            done = sum(c for c, _ in layers.values())
            total = sum(t for _, t in layers.values())
            print("\r" + prog.update(done, total, msg.get("status", "")), end="", flush=True)
    prog.finish(f"{name} ready")
    print(f"\n{name} ready")


if __name__ == "__main__":
    pull(sys.argv[1] if len(sys.argv) > 1 else "gemma3:4b")
