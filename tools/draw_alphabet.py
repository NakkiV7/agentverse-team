"""Draw A to Z with the arm and save its overhead image for a legibility check.

    .venv/bin/python tools/draw_alphabet.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from buddy import config as C
from buddy.game.marks import capital_letter, stroke_segments
from buddy.motion.controller import ArmController, Job
from buddy.motion.ik import PenIK
from buddy.perception.sheet import CameraModel, SheetTracker, check_lines, find_sheet
from buddy.sim.world import World


def alphabet_strokes():
    return [s for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            for s in capital_letter(c, (-64 + 19 * (i % 7), 37 - 32 * (i // 7)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RUNTIME / "alphabet_overhead.png")
    args = parser.parse_args()
    t0 = time.monotonic()
    world = World()
    try:
        cam = CameraModel(*world.camera_model())
        tracker = SheetTracker()
        ctl = ArmController(PenIK(world.model, world.pen.tip_body, world.pen.axis_body, World.ARM),
                            world.arm_q(), lambda kind, msg: print(f"{kind}: {msg}", flush=True))
        for t in (0.1, 0.2, 0.3):
            tracker.update(find_sheet(world.render_overhead(), cam), t)
        strokes = alphabet_strokes()
        job = ctl.submit(Job("alphabet", strokes_mm=strokes, sheet=lambda: tracker.track, home=True))
        forces = []
        for k in range(900 * C.CONTROL_HZ):
            world.command(ctl.tick(world.touch()))
            world.step(round(1 / C.CONTROL_HZ / C.TIMESTEP))
            forces.append(world.touch())
            if k % (C.CONTROL_HZ // C.CAMERA_HZ) == 0:
                tracker.update(find_sheet(world.render_overhead(), cam), .3 + (k + 1) / C.CONTROL_HZ)
            if job.done.is_set():
                break
        rgb = world.render_overhead()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not save {args.output}")
        if not job.done.is_set() or not job.ok:
            raise RuntimeError(job.error or "Alphabet drawing timed out")
        coverage = check_lines(rgb, cam, tracker.track.pose, stroke_segments(strokes))
        pressing = np.array(forces)[np.array(forces) > .05]
        print(f"26 letters; {len(strokes)} strokes; minimum segment coverage {min(coverage):.1%}; "
              f"median pressure {np.median(pressing):.2f} N; {time.monotonic() - t0:.1f} s wall; {args.output}")
        if min(coverage) < C.LINE_OK_COVERAGE:
            raise RuntimeError("Some letter strokes could not be verified")
    finally:
        world.close()


if __name__ == "__main__":
    main()
