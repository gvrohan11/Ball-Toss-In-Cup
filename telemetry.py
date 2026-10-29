"""Per-attempt telemetry: a timeline of what the arm did, stored with the attempt in Atlas.

Every snapshot records time since start, the 6 joint angles, the gripper pose in the
world frame and the finger position. Moves record planning+execution time. Key views
store a small JPEG thumbnail. Everything is best-effort: telemetry never stops a run.

Why: hit/miss alone can't explain a miss. With joints, poses, timing and the release
moment logged, misses can be traced (wrong cup estimate vs. arm didn't reach vs. slow
release), and the throw (Phase 2) needs exactly this data to fit release timing.

Stored on the attempt document as:
  telemetry.events[]   {t, stage, joints_deg[6], gripper_pose{x,y,z,o_x,o_y,o_z,theta}, finger_pos, ...}
  telemetry.moves[]    {stage, target, ok, seconds}
  telemetry.phases     {name: seconds}
  telemetry.images[]   {stage, jpeg (binary, ~320 px wide)}
  telemetry.config     {corrections, heights, speeds, git sha}
"""
import subprocess
import time

ARM, GRIPPER = "arm", "gripper"
THUMB_WIDTH = 320


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


class Telemetry:
    def __init__(self, robot, go):
        self.robot, self.go = robot, go
        self.t0 = time.monotonic()
        self.events, self.moves, self.images = [], [], []
        self.phases, self._phase = {}, None
        self.config = {"git": _git_sha()}

    def now(self):
        return round(time.monotonic() - self.t0, 3)

    # ---- phases -------------------------------------------------------------
    def phase(self, name):
        t = self.now()
        if self._phase:
            prev, start = self._phase
            self.phases[prev] = round(t - start, 2)
        self._phase = (name, t) if name else None

    # ---- snapshots ----------------------------------------------------------
    async def snap(self, stage, **extra):
        ev = {"t": self.now(), "stage": stage, **extra}
        if self.go and self.robot is not None:
            try:
                from viam.components.arm import Arm
                ev["joints_deg"] = [round(v, 2) for v in (await Arm.from_robot(self.robot, ARM)
                                                          .get_joint_positions(timeout=2)).values]
            except Exception as e:
                ev["joints_err"] = str(e)[:80]
            try:
                from viam.services.motion import MotionClient
                p = (await MotionClient.from_robot(self.robot, "builtin")
                     .get_pose(GRIPPER, "world", timeout=2)).pose
                ev["gripper_pose"] = {k: round(getattr(p, k), 2)
                                      for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}
            except Exception as e:
                ev["pose_err"] = str(e)[:80]
            try:
                from viam.components.gripper import Gripper
                ev["finger_pos"] = (await Gripper.from_robot(self.robot, GRIPPER)
                                    .do_command({"get": True}, timeout=2)).get("pos")
            except Exception:
                pass
        self.events.append(ev)
        return ev

    def move(self, stage, target, ok, seconds):
        self.moves.append({"t": self.now(), "stage": stage, "target": target,
                           "ok": ok, "seconds": round(seconds, 2)})

    async def thumbnail(self, stage):
        if not (self.go and self.robot is not None):
            return
        try:
            import cv2
            import numpy as np
            from viam.components.camera import Camera
            imgs, _ = await Camera.from_robot(self.robot, "cam").get_images(
                filter_source_names=["color"], timeout=3)
            frame = cv2.imdecode(np.frombuffer(imgs[0].data, np.uint8), cv2.IMREAD_COLOR)
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (THUMB_WIDTH, int(h * THUMB_WIDTH / w)))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.images.append({"t": self.now(), "stage": stage, "jpeg": buf.tobytes()})
        except Exception:
            pass

    def as_doc(self):
        self.phase(None)
        return {"events": self.events, "moves": self.moves, "phases": self.phases,
                "images": self.images, "config": self.config,
                "total_s": self.now()}
