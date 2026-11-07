"""Open the Buddy world in the interactive viewer. On macOS this MUST run under mjpython:

    uv run mjpython scripts/view_scene.py

Drag the sheet with double-click + Ctrl-drag to test "sheet moved". Press Space to pause.
"""
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("scenes/buddy.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
mujoco.viewer.launch(m, d)
