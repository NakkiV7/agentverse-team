"""Is this laptop ready? Run: uv run python scripts/check_env.py"""
import shutil
import subprocess
import sys

ok = True


def row(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"  {'OK ' if good else 'MISSING'}  {name:<28} {detail}")


print(f"python {sys.version.split()[0]}")
try:
    import mujoco
    row("mujoco", True, mujoco.__version__)
    m = mujoco.MjModel.from_xml_path("scenes/buddy.xml")
    row("scenes/buddy.xml loads", True, f"{m.nbody} bodies, {m.nu} actuators, {m.ncam} cameras")
    r = mujoco.Renderer(m, 48, 64); r.update_scene(mujoco.MjData(m)); r.render()
    row("offscreen rendering", True)
except Exception as e:  # noqa: BLE001
    row("mujoco", False, repr(e))
try:
    import cv2
    row("opencv", True, cv2.__version__)
except Exception as e:  # noqa: BLE001
    row("opencv", False, repr(e))
row("ollama binary", shutil.which("ollama"), shutil.which("ollama") or "brew install ollama")
try:
    import requests
    tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=2).json()
    names = [t["name"] for t in tags.get("models", [])]
    row("ollama server", True, ", ".join(names) or "no models yet: ollama pull gemma4")
    row("gemma4 pulled", any(n.startswith("gemma4") for n in names))
except Exception:  # noqa: BLE001
    row("ollama server", False, "start it: ollama serve  (or: brew services start ollama)")
row("whisper-cli (voice, later)", shutil.which("whisper-cli"), "brew install whisper-cpp")
print("\nall good" if ok else "\nsomething is missing, see above")
