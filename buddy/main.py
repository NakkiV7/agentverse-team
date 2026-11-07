"""One process, three layers. The sim (and its GL context) runs on the main thread, which macOS
requires for GLFW; every other module runs in its own thread and talks over the Bus.

    uv run python -m buddy.main                 # sim + perception + motion + agent + printing speaker + ui
    uv run python -m buddy.main --no-llm        # agent without Ollama
    uv run python -m buddy.main --seconds 10    # run for 10 s then exit (smoke test)
"""
from __future__ import annotations

import argparse
import math
import threading
import time

from buddy.agent.agent import Agent
from buddy.agent.llm import LLM
from buddy.bus import Bus
from buddy.motion.planner import Motion
from buddy.perception.perceive import Perception
from buddy.sim.world import World
from buddy.ui.server import UI
from buddy.voice.voice import Listener, Speaker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--seconds", type=float, default=0, help="run for this long then exit (0 = until Ctrl-C)")
    args = ap.parse_args()

    bus = Bus()
    world = World(bus)
    llm = None if args.no_llm else LLM()

    def sheet_pose():
        """Sheet pose for motion: from perception when it is confident, else straight from the sim."""
        p = bus.percept.get()
        if p and p.sheet and p.sheet.confidence > 0.5:
            return p.sheet.x, p.sheet.y, p.sheet.yaw
        x, y, _ = world.d.body("sheet").xpos
        w, _, _, z = world.d.body("sheet").xquat                   # yaw-only rotation assumed
        return float(x), float(y), 2 * math.atan2(z, w)

    modules = {
        "perception": Perception(bus).run,
        "motion": Motion(bus, world.m, sheet_pose).run,
        "agent": Agent(bus, llm).run,
        "speaker": Speaker(bus).run,
        "listener": Listener(bus, llm).run,
        "ui": UI(bus, world, llm).run,
    }
    threads = [threading.Thread(target=fn, name=name, daemon=True) for name, fn in modules.items()]

    def console():
        t0 = time.monotonic()
        frames = 0
        last = None
        while not bus.stop.is_set():
            for line in bus.transcript.drain():
                print(line, flush=True)
            f = bus.sensors.get()
            if f is not None and f.t != last:
                frames += 1
                last = f.t
            if args.seconds and time.monotonic() - t0 >= args.seconds:
                p = bus.percept.get()
                print(f"[main] {frames} camera frames in {args.seconds:g} s; percept={'yes' if p else 'none'}; "
                      f"threads alive: {[t.name for t in threads if t.is_alive()]}", flush=True)
                bus.stop.set()
            time.sleep(0.1)

    threads.append(threading.Thread(target=console, name="console", daemon=True))
    for t in threads:
        t.start()
    print("buddy running: sim (main thread), " + ", ".join(modules), flush=True)
    try:
        world.run()                      # blocks until bus.stop
    except KeyboardInterrupt:
        bus.stop.set()
    for t in threads:
        t.join(timeout=2)


if __name__ == "__main__":
    main()
