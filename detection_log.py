"""Detection log: every frame the detectors looked at and what they found, saved locally.

One folder per attempt (observations/ is git-ignored):
  observations/detections/<YYYYmmdd-HHMMSS>/
    log.jsonl                one line per frame: stage (e.g. cup_view_3), the view pose,
                             detector, every candidate (pixel box/circle, world mm, kept or
                             why not), plus survey/target events
    0007_cup_view_3.jpg      the raw camera frame (re-run cup_rim.py / ball_finder on it offline)
    0007_cup_view_3_det.jpg  the same frame with the candidates drawn on it
    0007_cup_view_3_depth.png  aligned depth (16-bit mm) of that frame, when YOLO ran:
                             replay YOLO's 3D positions offline (cup_detection.locate_cups)
  Cup survey views save images for ONE frame each (the first reading after the arm
  arrives); the other readings of that view are only lines in log.jsonl.

Detectors call record(); main.py calls start() per attempt and context() per stage.
With no log started, every call is a no-op. Best-effort: a logging problem prints once
and never stops a run.

    python detection_log.py                 # summary of the newest log folder
    python detection_log.py <folder>        # summary of a given folder
"""
import json
import math
import sys

import numpy as np
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "observations" / "detections"
JPEG_QUALITY = 85

# Colors (BGR) per candidate kind; rejected candidates are drawn in grey.
COLORS = {"rim": (0, 200, 0), "red_body": (255, 120, 0), "ball": (0, 165, 255),
          "yolo": (255, 0, 255)}
REJECTED = (150, 150, 150)

current = None   # DetectionLog of the running attempt


def _tidy(v):
    """Round floats (mm / px to 0.1, small values like confidences to 0.001) for readability."""
    if isinstance(v, dict):
        return {k: _tidy(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_tidy(x) for x in v]
    if hasattr(v, "item") and not hasattr(v, "shape") or getattr(v, "shape", None) == ():
        v = v.item()
    if isinstance(v, float):
        return round(v, 1) if abs(v) >= 10 else round(v, 3)   # mm/px to 0.1; scores/fractions kept
    return v


def _jsonable(v):
    if hasattr(v, "tolist"):          # numpy arrays and scalars
        return v.tolist()
    if isinstance(v, (set, tuple)):
        return list(v)
    return str(v)


def _found(items):
    """Distinct objects kept in one frame (a YOLO box on a rim cup counts once)."""
    return sum(1 for it in items if it.get("kept", True) and not it.get("duplicate_of"))


def annotate(frame, items, title=""):
    """Copy of the frame with each candidate's box/circle and world position drawn on it."""
    import cv2
    img = frame.copy()
    for it in items:
        color = COLORS.get(it.get("kind"), (255, 255, 255)) if it.get("kept", True) else REJECTED
        label = it.get("kind", "?")
        if it.get("confidence") is not None:
            label += f" {it['confidence']:.2f}"
        if it.get("duplicate_of"):
            label += f" ={it['duplicate_of']}"
        if it.get("world_mm") is not None:
            x, y, z = it["world_mm"]
            label += f" ({x:.0f},{y:.0f},{z:.0f})"
        if not it.get("kept", True):
            label += f" X {it.get('why', '')}"
        if it.get("box") is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in it["box"])
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            # YOLO labels go under the box so they don't sit on the rim box's label
            anchor = (x0, min(img.shape[0] - 5, y1 + 16)) if it.get("kind") == "yolo" else (x0, max(15, y0 - 6))
        elif it.get("circle") is not None:
            u, v, r = (int(round(c)) for c in it["circle"])
            cv2.circle(img, (u, v), r, color, 2)
            cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 10, 2)
            anchor = (u - r, max(15, v - r - 6))
        else:
            continue
        cv2.putText(img, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    kept = _found(items)
    cv2.putText(img, f"{title}  found {kept} ({len(items)} candidates)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


class DetectionLog:
    def __init__(self, root):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / stamp
        k = 1
        while self.dir.exists():
            k += 1
            self.dir = Path(root) / f"{stamp}-{k}"
        self.dir.mkdir(parents=True)
        self.n, self.ctx = 0, {"stage": "start"}
        self._max_frames, self._saved = None, 0

    def context(self, stage, max_frames=None, **meta):
        """New stage. max_frames: save images for only this many of its frames (the rest
        still get their log.jsonl line). None = every frame."""
        self.ctx = {"stage": stage, **meta}
        self._max_frames, self._saved = max_frames, 0

    def _write(self, line):
        with (self.dir / "log.jsonl").open("a") as f:
            f.write(json.dumps(_tidy(line), default=_jsonable) + "\n")

    def event(self, kind, **data):
        self._write({"t": round(time.time(), 3), "event": kind, **self.ctx, **data})

    def record(self, detector, frame, items, depth=None, **meta):
        import cv2
        self.n += 1
        name = f"{self.n:04d}_{self.ctx['stage']}"
        line = {"t": round(time.time(), 3), "n": self.n, **self.ctx, "detector": detector,
                "found": _found(items), "items": items, **meta}
        if frame is not None and (self._max_frames is None or self._saved < self._max_frames):
            self._saved += 1
            if depth is not None:
                cv2.imwrite(str(self.dir / f"{name}_depth.png"), np.asarray(depth, np.uint16))
                line["depth"] = f"{name}_depth.png"
            params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            cv2.imwrite(str(self.dir / f"{name}.jpg"), frame, params)
            cv2.imwrite(str(self.dir / f"{name}_det.jpg"),
                        annotate(frame, items, f"{self.ctx['stage']} {detector}"), params)
            line["image"], line["annotated"] = f"{name}.jpg", f"{name}_det.jpg"
        self._write(line)


_warned = False


def _safe(fn, *a, **k):
    global _warned
    if current is None:
        return
    try:
        fn(*a, **k)
    except Exception as e:
        if not _warned:
            print(f"  [detection log] not saving ({type(e).__name__}: {e})")
            _warned = True


def start(root=None):
    """New log folder for one attempt. Returns it, or None if it can't be created."""
    global current
    try:
        current = DetectionLog(root or ROOT)
        print(f"  detection log: {current.dir}")
    except Exception as e:
        print(f"  [detection log] disabled ({e})")
        current = None
    return current


def stop():
    global current
    current = None


def context(stage, max_frames=None, **meta):
    _safe(lambda: current.context(stage, max_frames, **meta))


def event(kind, **data):
    _safe(lambda: current.event(kind, **data))


def record(detector, frame, items, depth=None, **meta):
    """items: [{kind, box=[x0,y0,x1,y1] or circle=[u,v,r], world_mm, kept, why, ...}].
    depth: optional aligned depth (mm); saved once per stage."""
    _safe(lambda: current.record(detector, frame, items, depth, **meta))


def _xy(p):
    return f"({p[0]:.0f},{p[1]:+.0f})"


def _frame_line(l):
    def tag(i):
        if i.get("duplicate_of"):
            dx, dy = i["offset_vs_rim_mm"]
            return f"Y={i['duplicate_of']}(off {dx:+.0f},{dy:+.0f})"
        return ("Y" if i.get("kind") == "yolo" else "") + _xy(i["world_mm"]) + ("*" if i.get("chosen") else "")
    kept = [tag(i) for i in l["items"] if i.get("kept", True) and i.get("world_mm")]
    other = sum(1 for i in l["items"] if i.get("kept", True) and not i.get("world_mm"))
    why = [i.get("why", "?") for i in l["items"] if not i.get("kept", True)]
    return (f"  {l['n']:4d}  {l['stage']:<11} {l['detector']:<16} "
            + (" ".join(kept) or "-")
            + (f"  +{other} not used" if other else "")
            + (f"  rejected: {', '.join(why)}" if why else "")
            + ("  [img]" if l.get("annotated") else ""))


def _event_lines(l):
    e = l["event"]
    if e == "survey":
        out = [f"  SURVEY: {len(l['cups'])} cup(s) from {len(l.get('sightings', []))} sightings"]
        for n, c in enumerate(l["cups"], 1):
            x, y = c["pos_mm"][:2]
            far = "" if c.get("reachable", True) else "  <- too far to drop"
            out.append(f"     cup {n}: {_xy(c['pos_mm'])}  {math.hypot(x, y) / 10:.0f} cm out, "
                       f"seen {c['sightings']}x{far}")
        return out
    if e == "target":
        refused = f"  REFUSED: {l['refused']}" if l.get("refused") else ""
        return [f"  TARGET: cup {l['number']} of {l['of']} at {_xy(l['cup_mm'])}{refused}"]
    if e == "yolo_vs_rim":
        dx, dy = l["median_offset_mm"]
        return [f"  YOLO vs RIM: rim - YOLO = ({dx:+.0f}, {dy:+.0f}) mm over {l['n']} shared sightings"]
    if e == "result":
        verdict = "HIT" if l.get("hit") else "MISS"
        extra = l.get("failure", "")
        if l.get("miss_mm"):
            extra += f" (landed {l['miss_mm']['x'] / 10:+.0f} cm past, {l['miss_mm']['y'] / 10:+.0f} cm left)"
        return [f"  RESULT: {verdict} {extra}".rstrip()]
    return [f"  {e}: " + json.dumps({k: v for k, v in l.items() if k not in ("t", "event", "stage")})]


def summarize(folder):
    """Readable account of one attempt: frame by frame, then survey / target / result."""
    folder = Path(folder)
    lines = [json.loads(s) for s in (folder / "log.jsonl").read_text().splitlines() if s]
    frames = [l for l in lines if "detector" in l]
    out = [f"{folder}  ({len(frames)} frames; [img] = saved as NNNN_<stage>_det.jpg)",
           "     #  stage       detector         found (x,y mm; * = chosen ball; Y = YOLO, Y=rim = YOLO saw the same cup)"]
    for l in lines:
        out += [_frame_line(l)] if "detector" in l else _event_lines(l)
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        folders = sorted(p for p in ROOT.glob("*") if p.is_dir()) if ROOT.exists() else []
        if not folders:
            raise SystemExit(f"no detection logs in {ROOT}")
        folder = folders[-1]
    print(summarize(folder))
