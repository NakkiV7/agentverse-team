"""The simulated world: SO-101 holding a felt-tip pen, a blank A4 sheet on a table, an overhead RGB-D camera and a
touch sensor at the pen tip.

Ink. MuJoCo has no notion of ink, so the sheet carries its own texture and the world paints into it: on every
physics step where the pen-tip touch sensor reads a contact, the tip position is converted to sheet coordinates
and a line segment is drawn into the texture, which is then re-uploaded to the renderers. Because the ink lives in
the sheet's texture, it moves with the sheet when someone slides it.

Everything is built in code from the unmodified SO-101 model (assets/so101), so the robot description stays
exactly as published.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from .. import config as C

PEN_GROUP, WORLD_GROUP = 2, 1          # contact bits: pen tip <-> sheet/table only, robot links <-> table only


def _quat_z(yaw: float):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _site_frame(spec_model_xml):
    """Pose of the 'gripperframe' site in the gripper body (from the model), to place the pen between the jaws."""
    m = mujoco.MjModel.from_xml_path(str(spec_model_xml))
    sid = m.site("gripperframe").id
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.site_quat[sid])
    return m.site_pos[sid].copy(), R.reshape(3, 3)


@dataclass
class PenGeometry:
    tip_body: np.ndarray        # pen tip contact point in the gripper body frame (bottom of the felt tip)
    axis_body: np.ndarray       # unit pen axis pointing toward the tip, gripper body frame


def build_model():
    """Return (MjModel, PenGeometry)."""
    spec = mujoco.MjSpec.from_file(str(C.SO101_XML))
    spec.option.timestep = C.TIMESTEP
    W, H = C.OVERHEAD_CAM["size"]
    spec.visual.global_.offwidth = max(W, C.SCENE_VIEW["size"][0], 1280)
    spec.visual.global_.offheight = max(H, C.SCENE_VIEW["size"][1], 960)
    spec.visual.quality.shadowsize = 2048
    wb = spec.worldbody

    # lights and table
    wb.add_light(pos=[0.2, -0.3, 1.2], dir=[-0.1, 0.25, -1], diffuse=[0.75, 0.75, 0.72], castshadow=True)
    wb.add_light(pos=[0.5, 0.4, 0.9], dir=[-0.4, -0.4, -0.8], diffuse=[0.3, 0.3, 0.3], castshadow=False)
    tmat = spec.add_material(name="table")
    tmat.rgba = [0.62, 0.48, 0.35, 1]
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.45, 0.45, 0.01], pos=[0.2, 0, -0.01],
                material="table", contype=WORLD_GROUP | PEN_GROUP, conaffinity=WORLD_GROUP | PEN_GROUP)

    # the A4 sheet: a thin box mesh whose top face carries the ink texture (u along the long edge, v along the short)
    tw, th = int(C.SHEET_SIZE_MM[0] * C.TEX_PX_PER_MM), int(C.SHEET_SIZE_MM[1] * C.TEX_PX_PER_MM)
    spec.add_texture(name="ink", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     width=tw, height=th, rgb1=[0.97, 0.97, 0.95], rgb2=[0.97, 0.97, 0.95])
    smat = spec.add_material(name="paper")
    smat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ink"
    smat.texrepeat = [1, 1]
    smat.texuniform = False
    sx, sy = C.SHEET_SIZE_MM[0] / 2000, C.SHEET_SIZE_MM[1] / 2000
    z0, z1 = 0.0, C.SHEET_THICK
    verts = [[-sx, -sy, z1], [sx, -sy, z1], [sx, sy, z1], [-sx, sy, z1],
             [-sx, -sy, z0], [sx, -sy, z0], [sx, sy, z0], [-sx, sy, z0]]
    uv = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1], [0, 1]]
    faces = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
             [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
    spec.add_mesh(name="sheet", uservert=[c for v in verts for c in v], usertexcoord=[c for t in uv for c in t],
                  userface=[i for f in faces for i in f])
    x0, y0, yaw0 = C.SHEET_POSE0
    sheet = wb.add_body(name="sheet", mocap=True, pos=[x0, y0, 0.0], quat=_quat_z(yaw0))
    sheet.add_geom(name="sheet", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="sheet", material="paper",
                   contype=PEN_GROUP, conaffinity=PEN_GROUP, friction=[0.6, 0.005, 0.0001])

    # the pen, rigidly held between the jaws (in the 'gripper' body). Only the felt tip can touch anything.
    site_p, site_R = _site_frame(C.SO101_XML)
    fixed_tip = np.array([0.0063, 0.0, -0.0010])             # fixed fingertip in the gripperframe site (measured)

    def in_body(p_site):
        return site_p + site_R @ np.asarray(p_site, float)

    z = fixed_tip[2] + C.PEN_OFFSET_FROM_FIXED
    x_tip = fixed_tip[0] + C.PEN_PAST_TIPS                   # x = along the jaws, toward the fingertips
    r_tip = C.PEN_TIP_RADIUS
    g = spec.body("gripper")
    g.add_geom(name="pen_barrel", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[C.PEN_RADIUS, 0, 0],
               fromto=[*in_body([-0.065, 0, z]), *in_body([x_tip - 0.010, 0, z])],
               rgba=[0.15, 0.35, 0.80, 1], contype=0, conaffinity=0, mass=0.01)
    g.add_geom(name="pen_cone", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0024, 0, 0],
               fromto=[*in_body([x_tip - 0.010, 0, z]), *in_body([x_tip - r_tip, 0, z])],
               rgba=[0.9, 0.9, 0.9, 1], contype=0, conaffinity=0, mass=0.001)
    # spring-loaded felt tip: the tip slides up into the barrel against a spring (like a plotter pen holder), so a
    # few millimetres of height error only change the pressure a little instead of lifting off or ramming the paper
    axis_b = site_R[:, 0] / np.linalg.norm(site_R[:, 0])
    tip_body = g.add_body(name="pen_tip_body", pos=in_body([x_tip - r_tip, 0, z]))
    tip_body.add_joint(name="pen_spring", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis_b, range=[-0.006, 0.0003],
                       stiffness=C.PEN_SPRING_K, springref=0.0, damping=4.0, armature=0.02)
    tip_body.add_geom(name="pen_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r_tip, 0, 0], rgba=[0.05, 0.06, 0.25, 1],
                      contype=PEN_GROUP, conaffinity=PEN_GROUP, condim=3, friction=[0.25, 0.005, 0.0001], mass=0.002)
    tip_body.add_site(name="pen_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.003, 0, 0], rgba=[1, 0, 0, 0])
    spec.add_sensor(name="pen_touch", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="pen_tip")

    # overhead RGB-D camera: straight down, image up = toward the robot, image right = robot's left (player's view)
    cam = C.OVERHEAD_CAM
    wb.add_camera(name="overhead", pos=list(cam["pos"]), quat=_quat_z(math.pi / 2), fovy=cam["fovy"])

    model = spec.compile()
    pen = PenGeometry(tip_body=in_body([x_tip, 0, z]), axis_body=site_R[:, 0] / np.linalg.norm(site_R[:, 0]))
    return model, pen


class World:
    """Owns the MuJoCo model/data, the renderers and the ink. Call everything from ONE thread (the main thread on
    macOS: OpenGL contexts are not shared across threads)."""

    ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    def __init__(self):
        self.model, self.pen = build_model()
        m = self.model
        self.data = mujoco.MjData(m)
        self.lock = threading.Lock()
        self.qadr = np.array([m.joint(n).qposadr[0] for n in self.ARM])
        self.act = np.array([m.actuator(n).id for n in self.ARM])
        self.grip_act = m.actuator("gripper").id
        self.grip_qadr = m.joint("gripper").qposadr[0]
        self.gripper_body = m.body("gripper").id
        self.sheet_mocap = m.body("sheet").mocapid[0]
        self.touch_adr = m.sensor_adr[m.sensor("pen_touch").id]
        self.spring_qadr = m.joint("pen_spring").qposadr[0]
        self.tex_id = m.texture("ink").id
        self.tex_w, self.tex_h = int(m.tex_width[self.tex_id]), int(m.tex_height[self.tex_id])
        adr = int(m.tex_adr[self.tex_id])
        self.ink = m.tex_data[adr: adr + self.tex_w * self.tex_h * 3].reshape(self.tex_h, self.tex_w, 3)
        self.ink_dirty = True
        self.last_ink_px = None
        self.ink_events = 0
        # start at home with the gripper closed on the pen
        self.set_arm_qpos(np.array(C.HOME_Q))
        self.data.ctrl[self.act] = C.HOME_Q
        self.data.qpos[self.grip_qadr] = C.GRIPPER_HOLD_Q
        self.data.ctrl[self.grip_act] = C.GRIPPER_HOLD_Q
        mujoco.mj_forward(m, self.data)
        W, H = C.OVERHEAD_CAM["size"]
        self.r_overhead = mujoco.Renderer(m, height=H, width=W)
        self.r_depth = mujoco.Renderer(m, height=H, width=W)
        self.r_depth.enable_depth_rendering()
        SW, SH = C.SCENE_VIEW["size"]
        self.r_scene = mujoco.Renderer(m, height=SH, width=SW)
        self.scene_cam = mujoco.MjvCamera()
        sv = C.SCENE_VIEW
        self.scene_cam.lookat[:] = sv["lookat"]
        self.scene_cam.distance, self.scene_cam.azimuth, self.scene_cam.elevation = sv["distance"], sv["azimuth"], sv["elevation"]

    # ------------------------------------------------------------ arm
    def set_arm_qpos(self, q5):
        self.data.qpos[self.qadr] = q5
        self.data.qvel[:] = 0

    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def command(self, q5) -> None:
        lo, hi = self.model.actuator_ctrlrange[self.act].T
        self.data.ctrl[self.act] = np.clip(q5, lo, hi)

    def touch(self) -> float:
        return float(self.data.sensordata[self.touch_adr])

    def pen_compression_mm(self) -> float:
        return -1000.0 * float(self.data.qpos[self.spring_qadr])

    def pen_tip_world(self) -> np.ndarray:
        R = self.data.xmat[self.gripper_body].reshape(3, 3)
        return self.data.xpos[self.gripper_body] + R @ self.pen.tip_body

    # ------------------------------------------------------------ sheet
    def sheet_pose(self):
        """Ground truth (x, y, yaw) of the sheet; the robot itself only learns it through the camera."""
        p = self.data.mocap_pos[self.sheet_mocap]
        q = self.data.mocap_quat[self.sheet_mocap]
        return float(p[0]), float(p[1]), 2 * math.atan2(q[3], q[0])

    def move_sheet(self, x, y, yaw) -> None:
        self.data.mocap_pos[self.sheet_mocap] = [x, y, 0.0]
        self.data.mocap_quat[self.sheet_mocap] = _quat_z(yaw)

    def world_to_sheet_mm(self, p) -> np.ndarray:
        x, y, yaw = self.sheet_pose()
        dx, dy = p[0] - x, p[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy]) * 1000.0

    def _ink_px(self, sheet_mm):
        """Sheet mm -> texture pixel (texture rows run along the sheet's short edge)."""
        u = (sheet_mm[0] + C.SHEET_SIZE_MM[0] / 2) / C.SHEET_SIZE_MM[0]
        v = (sheet_mm[1] + C.SHEET_SIZE_MM[1] / 2) / C.SHEET_SIZE_MM[1]
        return (int(round(u * (self.tex_w - 1))), int(round(v * (self.tex_h - 1))))

    def paint(self, sheet_mm_from, sheet_mm_to, rgb=C.INK_RGB, radius_mm=C.INK_RADIUS_MM) -> None:
        """Paint a segment of ink on the sheet (sheet frame, mm). Used by the pen and by the human player."""
        a, b = self._ink_px(sheet_mm_from), self._ink_px(sheet_mm_to)
        thick = max(1, int(round(2 * radius_mm * C.TEX_PX_PER_MM)))
        cv2.line(self.ink, a, b, tuple(int(c) for c in rgb), thick, cv2.LINE_AA)
        self.ink_dirty = True

    def clear_ink(self) -> None:
        self.ink[:] = (247, 247, 242)
        self.ink_dirty = True
        self.last_ink_px = None

    def _pen_ink(self) -> None:
        if self.touch() > C.TOUCH_FIND * 0.5:
            mm = self.world_to_sheet_mm(self.pen_tip_world())
            inside = abs(mm[0]) < C.SHEET_SIZE_MM[0] / 2 and abs(mm[1]) < C.SHEET_SIZE_MM[1] / 2
            if inside:
                self.paint(self.last_ink_px if self.last_ink_px is not None else mm, mm)
                self.last_ink_px = mm
                self.ink_events += 1
                return
        self.last_ink_px = None

    # ------------------------------------------------------------ time
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._pen_ink()

    # ------------------------------------------------------------ rendering
    def _upload_ink(self, renderer) -> None:
        renderer._gl_context.make_current()
        mujoco.mjr_uploadTexture(self.model, renderer._mjr_context, self.tex_id)

    def render_overhead(self):
        """(rgb uint8 HxWx3, depth float32 HxW in metres) from the overhead camera."""
        if self.ink_dirty:
            for r in (self.r_overhead, self.r_scene):
                self._upload_ink(r)
            self.ink_dirty = False
        self.r_overhead.update_scene(self.data, camera="overhead")
        rgb = self.r_overhead.render().copy()
        self.r_depth.update_scene(self.data, camera="overhead")
        depth = self.r_depth.render().copy()
        return rgb, depth

    def render_scene(self):
        if self.ink_dirty:
            for r in (self.r_overhead, self.r_scene):
                self._upload_ink(r)
            self.ink_dirty = False
        self.r_scene.update_scene(self.data, camera=self.scene_cam)
        return self.r_scene.render().copy()

    def camera_model(self):
        """Overhead camera intrinsics/extrinsics, as a real system would get them from a calibration."""
        m, d = self.model, self.data
        cid = m.camera("overhead").id
        W, H = C.OVERHEAD_CAM["size"]
        f = (H / 2) / math.tan(math.radians(m.cam_fovy[cid]) / 2)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        R_mj = d.cam_xmat[cid].reshape(3, 3)          # MuJoCo camera: x right, y up, looks along -z
        R_cv = R_mj @ np.diag([1.0, -1.0, -1.0])      # OpenCV camera: x right, y down, z forward
        return K, R_cv, d.cam_xpos[cid].copy()

    def close(self):
        for r in (self.r_overhead, self.r_depth, self.r_scene):
            r.close()
