"""Hour-1 check: a scripted line of ink on the sheet, seen by the overhead camera.

    uv run python scripts/render_overhead.py            -> out/overhead.png, out/side.png, out/depth.png
"""
import pathlib

import mujoco
import numpy as np
import PIL.Image

from buddy.motion.planner import grid_strokes, x_strokes
from buddy.sim.ink import Ink

out = pathlib.Path("out"); out.mkdir(exist_ok=True)
m = mujoco.MjModel.from_xml_path("scenes/buddy.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
for _ in range(300):
    mujoco.mj_step(m, d)

r = mujoco.Renderer(m, 480, 640)
ink = Ink(m, r)
for s in grid_strokes() + x_strokes(4):
    for p, q in zip(s.points_mm, s.points_mm[1:]):
        ink.line(p, q, radius_mm=1.0)
ink.line((-120, -80), (-120, 80), radius_mm=1.0, rgb=(200, 30, 30))   # red line on the robot's side (x = -120 mm)
ink.flush()

for cam in ("overhead", "side", "front"):
    r.update_scene(d, camera=cam)
    PIL.Image.fromarray(r.render()).save(out / f"{cam}.png")
r.enable_depth_rendering(); r.update_scene(d, camera="overhead")
depth = r.render()
PIL.Image.fromarray((255 * np.clip((0.75 - depth) / 0.3, 0, 1)).astype(np.uint8)).save(out / "depth.png")
print("wrote", sorted(p.name for p in out.iterdir()))
