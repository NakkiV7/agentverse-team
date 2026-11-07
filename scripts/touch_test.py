"""Hour-3 check, first half: IK takes the pen to the sheet, touches down, drags a line; ink appears.

    uv run python scripts/touch_test.py        -> prints touch force / pen compression, writes out/touch_*.png
"""
import pathlib
import time

import mujoco
import numpy as np
import PIL.Image

from buddy.bus import Bus
from buddy.contracts import JointTargets
from buddy.motion.ik import PenIK
from buddy.sim.ink import sheet_mm_to_world, world_to_sheet_mm
from buddy.sim.world import World

out = pathlib.Path("out"); out.mkdir(exist_ok=True)
bus = Bus()
w = World(bus)
w.renderer = mujoco.Renderer(w.m, 480, 640)
from buddy.sim.ink import Ink
w.ink = Ink(w.m, w.renderer)
ik = PenIK(w.m)

def settle(n):
    for _ in range(n):
        w.step()

def goto(target_world, label, ticks=300):
    q5, ok, ep, ea = ik.solve(target_world, w.q()[:5])
    bus.joint_targets.set(JointTargets(np.concatenate([q5, [0.3]])))
    settle(ticks)
    tip = w.pen_tip()
    x, y, z = world_to_sheet_mm(w.m, w.d, tip)
    print(f"{label:<22} ik_ok={ok} ik_err={ep*1000:.1f}mm axis={np.degrees(ea):.1f}deg | tip sheet-frame "
          f"x={x:6.1f} y={y:6.1f} z={z:5.1f} mm  touch={w.touch_force():.2f} N  compress={w.pen_compress()*1000:.1f} mm")
    return ok

settle(200)
t0 = time.perf_counter()
goto(sheet_mm_to_world(w.m, w.d, 0, 0, 0.03), "hover centre")
goto(sheet_mm_to_world(w.m, w.d, 0, 0, -0.004), "press centre")
# drag a 60 mm line toward the robot's left (+y) while pressed
for yy in np.linspace(0, 60, 13):
    goto(sheet_mm_to_world(w.m, w.d, 0, yy, -0.004), f"drag y={yy:.0f}", ticks=40)
goto(sheet_mm_to_world(w.m, w.d, 0, 60, 0.03), "lift")
print(f"total sim wall time {time.perf_counter()-t0:.1f}s; ink pixels: {(w.ink.tex[:,:,0] < 128).sum()}")
for cam in ("overhead", "side"):
    w.ink.flush(); w.renderer.disable_depth_rendering(); w.renderer.update_scene(w.d, camera=cam)
    PIL.Image.fromarray(w.renderer.render()).save(out / f"touch_{cam}.png")
print("wrote out/touch_overhead.png out/touch_side.png")
