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

    # the player's hand holding their own pen (mocap, visual only). Its origin is the pen tip; the forearm comes
    # from the player's side of the table (+x). The depth camera sees it like a real hand.
    skin = [0.87, 0.68, 0.56, 1]
    hand = wb.add_body(name="player_hand", mocap=True, pos=list(C.HAND_PARK))
    hand.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0045, 0, 0], fromto=[0, 0, 0.002, 0.035, 0.012, 0.115],
                  rgba=[0.12, 0.12, 0.14, 1], contype=0, conaffinity=0)
    hand.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.045, 0.032, 0.022], pos=[0.06, 0.02, 0.125],
                  rgba=skin, contype=0, conaffinity=0)
    hand.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.008, 0, 0], fromto=[0.03, 0.0, 0.10, 0.012, 0.004, 0.035],
                  rgba=skin, contype=0, conaffinity=0)                       # index finger along the pen
    hand.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.030, 0, 0], fromto=[0.10, 0.03, 0.14, 0.32, 0.09, 0.21],
                  rgba=[0.35, 0.45, 0.65, 1], contype=0, conaffinity=0)     # forearm in a sleeve

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
        dW, dH = C.DEPTH_SIZE
        self.r_depth = mujoco.Renderer(m, height=dH, width=dW)
        self.r_depth.enable_depth_rendering()
        self.r_seg = mujoco.Renderer(m, height=dH, width=dW)
        self.r_seg.enable_segmentation_rendering()
        self.rng = np.random.default_rng(7)
        self.noise = C.CAMERA_NOISE
        # which geoms belong to the robot (the arm knows its own body: used to filter itself out of the depth image)
        robot_bodies = set()
        for b in range(m.nbody):
            r = b
            while r > 0 and m.body(r).name != "base":
                r = m.body_parentid[r]
            if r > 0:
                robot_bodies.add(b)
        self.robot_geom = np.array([m.geom_bodyid[g] in robot_bodies for g in range(m.ngeom)] + [False])
        self.hand_mocap = m.body("player_hand").mocapid[0]
        self.anim = []                                   # running animations (sheet slides, the player's hand)
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

    # ------------------------------------------------------------ animations (scene changes made by "the player")
    def slide_sheet(self, dx, dy, dyaw, duration=1.2):
        x, y, yaw = self.sheet_pose()
        self.anim.append({"kind": "sheet", "t": 0.0, "T": duration, "p0": (x, y, yaw), "p1": (x + dx, y + dy, yaw + dyaw)})

    def hand_strokes(self, strokes_mm, rgb=C.PLAYER_INK_RGB, speed=0.12):
        """The player's hand draws these polylines (sheet mm) on the sheet: approach, pen down, draw, lift, leave."""
        self.anim.append({"kind": "strokes", "t": 0.0, "strokes": [np.asarray(s, float) for s in strokes_mm],
                          "rgb": rgb, "speed": speed, "path": None})

    def reach_in(self, duration=4.0):
        """A hand reaches over the drawing area, stays, and leaves (to test the safety stop)."""
        self.anim.append({"kind": "reach", "t": 0.0, "T": duration})

    def busy_animating(self) -> bool:
        return bool(self.anim)

    def _hand_at(self, p):
        self.data.mocap_pos[self.hand_mocap] = p

    def _sheet_world(self, mm, z=C.SHEET_THICK):
        x, y, yaw = self.sheet_pose()
        c, s = math.cos(yaw), math.sin(yaw)
        u, v = mm[0] / 1000.0, mm[1] / 1000.0
        return np.array([x + c * u - s * v, y + s * u + c * v, z])

    def _hand_path(self, a):
        """Whole path of the player's pen tip: from rest, over each stroke (lifted between strokes), back to rest."""
        pts, ink = [np.array(C.HAND_PARK)], [False]
        for st in a["strokes"]:
            pts += [self._sheet_world(st[0], 0.02), self._sheet_world(st[0])]
            ink += [False, False]
            for q in st[1:]:
                pts.append(self._sheet_world(q))
                ink.append(True)
            pts.append(self._sheet_world(st[-1], 0.02))
            ink.append(False)
        pts.append(np.array(C.HAND_PARK))
        ink.append(False)
        P, ink = np.array(pts), np.array(ink)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        seg = np.where(ink[1:], seg, seg * 0.35)                         # moves in the air are quicker
        a["path"], a["ink"], a["L"], a["last_mm"] = P, ink, np.concatenate([[0], np.cumsum(seg)]), None

    def _animate(self, dt: float):
        if not self.anim:
            return
        a = self.anim[0]
        a["t"] += dt
        t = a["t"]
        if a["kind"] == "sheet":
            u = min(1.0, t / a["T"])
            u = u * u * (3 - 2 * u)
            self.move_sheet(*(a["p0"][i] + (a["p1"][i] - a["p0"][i]) * u for i in range(3)))
            done = t >= a["T"]
        elif a["kind"] == "reach":
            T = a["T"]
            centre = self._sheet_world((0.0, 0.0), 0.06)
            park = np.array(C.HAND_PARK)
            u = min(1.0, t / 0.8) if t < T - 0.8 else max(0.0, (T - t) / 0.8)
            self._hand_at(park + (centre - park) * (u * u * (3 - 2 * u)))
            done = t >= T
        else:                                                            # the player's hand drawing strokes
            if a["path"] is None:
                self._hand_path(a)
            L = a["L"]
            s_now = min(L[-1], t * a["speed"])
            i = min(int(np.searchsorted(L, s_now, side="right") - 1), len(a["path"]) - 2)
            w = (s_now - L[i]) / max(1e-9, L[i + 1] - L[i])
            p = a["path"][i] + (a["path"][i + 1] - a["path"][i]) * w
            self._hand_at(p)
            if a["ink"][i + 1]:
                mm = self.world_to_sheet_mm(p)
                if a["last_mm"] is not None:
                    self.paint(a["last_mm"], mm, rgb=a["rgb"], radius_mm=0.8)
                a["last_mm"] = mm
            else:
                a["last_mm"] = None
            done = s_now >= L[-1]
        if done:
            if a["kind"] != "sheet":
                self._hand_at(np.array(C.HAND_PARK))
            self.anim.pop(0)

    # ------------------------------------------------------------ time
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._pen_ink()
        self._animate(n * self.model.opt.timestep)

    # ------------------------------------------------------------ rendering
    def _upload_ink(self, renderer) -> None:
        renderer._gl_context.make_current()
        mujoco.mjr_uploadTexture(self.model, renderer._mjr_context, self.tex_id)

    def render_overhead(self):
        """Colour image from the overhead camera, with sensor noise and a slight lens blur."""
        if self.ink_dirty:
            for r in (self.r_overhead, self.r_scene):
                self._upload_ink(r)
            self.ink_dirty = False
        self.r_overhead.update_scene(self.data, camera="overhead")
        rgb = self.r_overhead.render().astype(np.float32)
        rgb = cv2.GaussianBlur(rgb, (3, 3), 0.7)
        rgb += self.rng.normal(0.0, self.noise, rgb.shape).astype(np.float32)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def render_depth(self):
        """(depth in metres, robot_mask) at DEPTH_SIZE from the same camera. robot_mask marks pixels showing the
        robot's own body, from a segmentation render: the arm filters itself out of what it considers an obstacle."""
        self.r_depth.update_scene(self.data, camera="overhead")
        depth = self.r_depth.render().copy()
        self.r_seg.update_scene(self.data, camera="overhead")
        seg = self.r_seg.render()
        ids = seg[:, :, 0]
        robot = self.robot_geom[np.where((ids >= 0) & (ids < len(self.robot_geom) - 1), ids, -1)]
        return depth, robot

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
        for r in (self.r_overhead, self.r_depth, self.r_seg, self.r_scene):
            r.close()
