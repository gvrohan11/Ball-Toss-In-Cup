"""Live view: what the robot sees and what it decides, while main.py runs.

Read-only. Follows the detection log main.py already writes (observations/detections/<time>/:
log.jsonl + the NNNN_<stage>_det.jpg frames) and serves one page:
  left   the latest camera frame with every candidate drawn (kept in colour, rejected grey)
  right  where the run is (survey -> target -> find ball -> close-up -> grab -> drop -> result),
         what it plans to do next, and every decision in plain words: what it saw, what it
         kept, what it rejected and why
Nothing here talks to the robot, imports robot code or writes a file. Standard library only.

    python live_view.py                    # http://127.0.0.1:8766, follows the newest log folder
    python live_view.py --port 9000
    # optional: also show the steps main.py prints (-u = unbuffered, so lines appear at once)
    python -u main.py --go 2>&1 | tee observations/live.txt

Robot code running on another machine over SSH: start this there, then on your laptop
    ssh -L 8766:localhost:8766 <user>@<machine>
and open http://localhost:8766 on the laptop.
"""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent / "observations" / "detections"
NARRATION = ROOT.parent / "live.txt"   # main.py output, when it is tee'd there
NARRATION_LINES = 60
MAX_FOLDERS = 30                       # how many past attempts the replay list offers
IMAGE_NAME = re.compile(r"^[\w.-]+\.jpg$")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def folders():
    """Log folders, newest first (names are timestamps, so they sort by time)."""
    if not ROOT.is_dir():
        return []
    return sorted((p.name for p in ROOT.iterdir() if (p / "log.jsonl").is_file()), reverse=True)


def read_lines(folder, start=0):
    """log.jsonl lines from index `start` on. A half-written last line waits for the next poll."""
    try:
        text = (ROOT / folder / "log.jsonl").read_text()
    except OSError:
        return []
    out = []
    for s in text.split("\n")[:-1][start:]:
        try:
            out.append(json.loads(s))
        except ValueError:
            out.append({"event": "unreadable"})
    return out


def narration(path):
    """Last lines main.py printed (when tee'd to `path`), or None."""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 32000))
            text = f.read().decode("utf-8", "replace")
        mtime = path.stat().st_mtime
    except OSError:
        return None
    lines = [ANSI.sub("", s).rstrip() for s in text.replace("\r", "\n").split("\n")]
    return {"lines": [s for s in lines if s.strip()][-NARRATION_LINES:], "mtime": mtime}


def image_path(folder, name):
    """Path of a frame inside a log folder, or None (no other file is ever served)."""
    if folder not in folders() or not IMAGE_NAME.match(name or ""):
        return None
    p = ROOT / folder / name
    return p if p.is_file() else None


def state(query, narration_file):
    """New log lines for the page. `have` = folder the page already shows, `from` = lines it has."""
    names = folders()
    want = query.get("folder", [""])[0]
    folder = want if want in names else (names[0] if names else None)
    start = int(query.get("from", ["0"])[0] or 0) if query.get("have", [""])[0] == folder else 0
    lines = read_lines(folder, start) if folder else []
    return {"folders": names[:MAX_FOLDERS], "folder": folder, "start": start, "lines": lines,
            "narration": narration(narration_file) if narration_file else None}


class Handler(BaseHTTPRequestHandler):
    narration_file = NARRATION

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            try:
                body = json.dumps(state(q, self.narration_file)).encode()
            except ValueError:
                return self._send(400, b"bad query", "text/plain")
            self._send(200, body, "application/json")
        elif url.path == "/img":
            p = image_path(q.get("folder", [""])[0], q.get("name", [""])[0])
            if p is None:
                return self._send(404, b"no such frame", "text/plain")
            self._send(200, p.read_bytes(), "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args):   # quiet: one line per poll would bury main.py's output
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot Live View</title>
<style>
:root{
  color-scheme:dark;
  --bg:#151716; --card:#1e2120; --card2:#252927; --line:#323735;
  --ink:#ecefed; --ink2:#b5bcb8; --ink3:#828a86;
  --ball:#ff8f3d; --cup:#ff5c70; --rim:#3ccf5a; --yolo:#e05cff; --ok:#4cc77f; --warn:#f2b33d; --bad:#ef6a5a;
  --mono:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;padding:12px 16px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:18px;font-weight:700}
h1 span{color:var(--ball)}
.status{display:flex;align-items:center;gap:8px;color:var(--ink2);font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--ink3)}
.dot.live{background:var(--ok);box-shadow:0 0 0 3px rgba(76,199,127,.2)}
.dot.idle{background:var(--warn)}
header .right{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
select,button{background:var(--card2);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:5px 9px;font:inherit;font-size:13px;cursor:pointer}
button:hover,select:hover{border-color:var(--ink3)}
button:focus-visible,select:focus-visible,.feed li[data-img]:focus-visible{outline:2px solid var(--ball);outline-offset:1px}
main{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(320px,1fr);gap:14px;padding:14px 16px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.label{font:600 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);margin:0 0 8px}
/* camera */
.frame{position:relative;background:#000;border-radius:8px;overflow:hidden;aspect-ratio:16/9;display:grid;place-items:center}
.frame img{width:100%;height:100%;object-fit:contain;display:block}
.frame .empty{color:var(--ink3);padding:20px;text-align:center}
.pin{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.65);color:var(--warn);font-size:12px;padding:3px 8px;border-radius:6px}
.caption{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline;margin:10px 0 0}
.caption b{font-size:15px}
.caption .meta{color:var(--ink2);font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2);margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.toggle{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--ink2)}
.strip{display:flex;gap:6px;overflow-x:auto;margin-top:12px;padding-bottom:4px}
.strip button{padding:0;border-radius:6px;overflow:hidden;flex:0 0 auto;width:96px;position:relative;border:2px solid transparent;background:#000}
.strip button.on{border-color:var(--ball)}
.strip img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
.strip span{position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);font:10px var(--mono);color:var(--ink2);padding:1px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.found{margin-top:12px;display:grid;gap:6px}
.found div{display:grid;grid-template-columns:14px minmax(0,1fr);gap:8px;font-size:13px}
.found i{width:10px;height:10px;border-radius:50%;margin-top:5px}
.found .no{color:var(--ink3)}
/* thinking */
.side{display:grid;gap:14px;min-width:0}
.steps{list-style:none;margin:0;padding:0;display:grid;gap:2px}
.steps li{display:grid;grid-template-columns:22px minmax(0,1fr);gap:8px;align-items:center;color:var(--ink3);padding:3px 0}
.steps li::before{content:"";width:12px;height:12px;border-radius:50%;border:2px solid var(--line);justify-self:center}
.steps li.done{color:var(--ink2)} .steps li.done::before{background:var(--ok);border-color:var(--ok)}
.steps li.now{color:var(--ink);font-weight:600} .steps li.now::before{border-color:var(--ball);background:var(--ball);box-shadow:0 0 0 4px rgba(255,143,61,.2)}
.steps small{font-weight:400;color:var(--ink3);margin-left:6px}
.plan{font-size:15px;line-height:1.55}
.plan .now{color:var(--ink)} .plan .next{color:var(--ink2);margin-top:8px;padding-top:8px;border-top:1px dashed var(--line)}
.plan .next b{color:var(--ball);font-weight:600}
.feed{list-style:none;margin:0;padding:0;display:grid;gap:8px;max-height:56vh;overflow-y:auto}
.feed li{background:var(--card2);border:1px solid var(--line);border-radius:8px;padding:8px 10px;min-width:0}
.feed li[data-img]{cursor:pointer} .feed li[data-img]:hover{border-color:var(--ink3)}
.feed li.sel{border-color:var(--ball)}
.feed .top{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.feed time{font:11px var(--mono);color:var(--ink3)}
.chip{font:600 10.5px var(--mono);padding:1px 6px;border-radius:4px;background:var(--line);color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
.chip.cup{background:rgba(255,92,112,.18);color:var(--cup)} .chip.ball{background:rgba(255,143,61,.18);color:var(--ball)}
.chip.ok{background:rgba(76,199,127,.18);color:var(--ok)} .chip.bad{background:rgba(239,106,90,.18);color:var(--bad)}
.feed .head{font-weight:600;margin-top:2px}
.feed ul{margin:4px 0 0;padding-left:18px;color:var(--ink2);font-size:13px}
.feed .rej{color:var(--ink3)}
.feed .cam{font-size:11px;color:var(--ink3)}
pre.console{margin:0;max-height:30vh;overflow:auto;font:12px/1.45 var(--mono);color:var(--ink2);white-space:pre-wrap;word-break:break-word}
details summary{cursor:pointer;list-style:none} details summary::-webkit-details-marker{display:none}
details summary .label{display:inline}
.hint{color:var(--ink3);font-size:12.5px;margin:6px 0 0}
code{font-family:var(--mono);font-size:.92em;background:var(--card2);padding:1px 4px;border-radius:4px}
@media (max-width:900px){main{grid-template-columns:minmax(0,1fr)}.feed{max-height:none}}
</style></head>
<body>
<header>
  <h1>Robot <span>live view</span></h1>
  <div class="status"><span class="dot" id="dot"></span><span id="status">connecting…</span></div>
  <div class="right">
    <label class="toggle"><input type="checkbox" id="drawn" checked> show detections</label>
    <select id="folder" aria-label="Attempt"></select>
    <button id="live" hidden>Back to live frame</button>
  </div>
</header>
<main>
  <section class="card">
    <p class="label">What the camera sees</p>
    <div class="frame" id="frame"><div class="empty">Waiting for the first camera frame…</div></div>
    <div class="caption" id="caption"></div>
    <div class="legend">
      <span><i style="background:var(--rim)"></i>cup (rim method)</span>
      <span><i style="background:var(--yolo)"></i>cup (YOLO)</span>
      <span><i style="background:var(--ball)"></i>ball</span>
      <span><i style="background:#3d8bff"></i>cup (red body fallback)</span>
      <span><i style="background:#969696"></i>rejected (reason drawn)</span>
    </div>
    <div class="found" id="found"></div>
    <div class="strip" id="strip"></div>
  </section>
  <div class="side">
    <section class="card">
      <p class="label">Where it is</p>
      <ol class="steps" id="steps"></ol>
    </section>
    <section class="card">
      <p class="label">Thinking</p>
      <div class="plan" id="plan"></div>
    </section>
    <section class="card">
      <p class="label">Decisions, newest first</p>
      <ul class="feed" id="feed"></ul>
    </section>
    <section class="card" id="consoleCard" hidden>
      <details open><summary><span class="label">What main.py printed</span></summary>
        <pre class="console" id="console"></pre></details>
    </section>
    <p class="hint" id="consoleHint" hidden>To also see every step main.py prints, run it as
      <code>python -u main.py --go 2>&amp;1 | tee observations/live.txt</code></p>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const S = {folder: null, lines: [], pick: "", pinned: null, lastNew: 0};
const PHASES = [
  ["survey", "Survey the cups", "7 views × 3 frames"],
  ["choose", "Choose a target cup", "nearest within 95 cm"],
  ["find", "Find a ball", "coarse look, search"],
  ["closeup", "Measure it up close", "camera over the ball"],
  ["grab", "Grab", "slow, 15 °/s"],
  ["drop", "Carry and drop", "not logged, inferred"],
  ["result", "Result", "hit / miss"],
];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const xy = p => p ? `(${Math.round(p[0])}, ${Math.round(p[1]) >= 0 ? "+" : ""}${Math.round(p[1])})` : "?";
const cm = p => p ? `${Math.round(Math.hypot(p[0], p[1]) / 10)} cm out` : "";
const clock = t => t ? new Date(t * 1000).toLocaleTimeString([], {hour12: false}) : "";
const img = (name, folder) => `/img?folder=${encodeURIComponent(folder || S.folder)}&name=${encodeURIComponent(name)}`;
const isCup = l => (l.detector || "").startsWith("cup_rim") || (l.detector || "").includes("detector");
const isBall = l => (l.detector || "").startsWith("ball_");

function viewName(v) {
  if (!v) return "";
  if (Array.isArray(v)) return `camera down, gripper at (${v.map(Math.round).join(", ")})`;   // scan_table.py views
  const aim = v.o_z <= -0.99 ? "straight down" : v.o_x < 0.6 ? "tilted 30°" : "tilted 45°";
  const side = v.y > 50 ? ", slid left" : v.y < -50 ? ", slid right" : "";
  return `${aim}${side}, z ${Math.round(v.z)}`;
}

function rejections(items) {
  const c = {};
  for (const it of items || []) if (it.kept === false) {
    const k = `${it.kind}: ${it.why || "?"}`; c[k] = (c[k] || 0) + 1;
  }
  return Object.entries(c).map(([k, n]) => n > 1 ? `${k} ×${n}` : k);
}

function phaseOf(l) {
  const st = l.stage || "", e = l.event;
  if (e === "result" || st === "result" || e === "session") return "result";
  if (e === "grab") return l.outcome === "closed on something" ? "drop" : "grab";
  if (st.startsWith("pick_retry")) return "grab";
  if (e === "closeup" || (isBall(l) && l.closeup)) return "closeup";
  if (st === "find_ball" || st.startsWith("ball_search")) return "find";
  if (e === "target" || st === "survey") return "choose";
  if (st.startsWith("cup_view")) return "survey";
  return null;
}

// One decision in plain words: {chip, cls, head, more: [], rej: []}
function describe(l, readingNo) {
  const it = l.items || [];
  if (isCup(l)) {
    const i = (l.stage || "").replace("cup_view_", "");
    const cups = it.filter(c => c.kept !== false && c.world_mm && !c.duplicate_of);
    const more = cups.map(c => `cup at ${xy(c.world_mm)}, ${cm(c.world_mm)}, by ${c.kind === "yolo" ? "YOLO only (+ its calibration)" : c.kind === "rim" ? "the pink rim" : "the red body"}` + (c.beyond_reach ? ", beyond drop reach" : ""));
    for (const d of it.filter(c => c.duplicate_of)) more.push(`YOLO saw the same cup (it was ${Math.round(Math.hypot(...d.offset_vs_rim_mm))} mm off the rim position)`);
    return {chip: `view ${i}/7`, cls: "cup", head: `${cups.length ? cups.length + " cup" + (cups.length > 1 ? "s" : "") : "No cups"} in this frame (reading ${readingNo} of 3)`,
            sub: viewName(l.view), more, rej: rejections(it)};
  }
  if (isBall(l)) {
    const st = l.stage || "";
    const where = l.closeup ? (st.startsWith("pick_retry") ? "Looking again after a miss" : "Close-up, camera above the ball")
                : st.startsWith("ball_search") ? `Searching the table, view ${st.replace("ball_search_", "")}`
                : st === "find_ball" ? "Coarse look for a ball" : "Scanning for balls";
    const kept = it.filter(b => b.kept !== false);
    const ch = it.find(b => b.chosen);
    const more = [];
    if (ch) more.push(`chose the ball at ${xy(ch.world_mm)}: ` + (l.closeup ? "the one nearest the image centre" : "the one with the most room around it for the fingers"));
    for (const b of kept) {
      const how = {edge_fit: "circle fit to its edge", split: "split from a group of touching balls", yolo_mask: "found by YOLO only"}[b.method] || b.method || "shape fit";
      const extra = [how, b.yolo != null && b.method !== "yolo_mask" ? `YOLO agrees ${b.yolo}` : null, b.sharpness != null && b.sharpness < 30 ? "blurry" : null].filter(Boolean).join(", ");
      if (b !== ch) more.push(`ball${b.world_mm ? " at " + xy(b.world_mm) : ""} (${extra})`); else more[0] += ` (${extra})`;
    }
    if (!kept.length) more.push(l.closeup ? "no ball near the image centre" : "no ball the fingers can take here");
    return {chip: l.closeup ? "close-up" : "ball", cls: "ball",
            head: `${where}: ${kept.length} ball${kept.length === 1 ? "" : "s"}` + (l.expected_r_px ? ` (a ball should look ${l.expected_r_px} px in radius from this height)` : ""),
            more, rej: rejections(it)};
  }
  switch (l.event) {
    case "survey": return {chip: "survey", cls: "cup", head: `Survey done: ${(l.sightings || []).length} sightings → ${(l.cups || []).length} distinct cup(s)`,
      more: ["sightings within 5 cm are one cup; a cup needs ≥ 3 sightings", ...(l.cups || []).map((c, n) => `cup ${n + 1}: ${xy(c.pos_mm)}, ${cm(c.pos_mm)}, seen ${c.sightings}×` + (c.reachable === false ? " (out of reach)" : ""))]};
    case "discarded": return {chip: "discard", cls: "bad", head: `Cup ${l.number} at ${xy(l.cup_mm)} discarded`, more: [l.why, `${l.left} cup(s) left`]};
    case "yolo_vs_rim": return {chip: "calib", head: `YOLO vs rim on ${l.n} shared sightings: rim − YOLO = ${xy(l.median_offset_mm)} mm`, more: ["the YOLO-only correction is (−54, +22); a big difference means it needs re-measuring"]};
    case "target": return l.refused
      ? {chip: "target", cls: "bad", head: `Cup ${l.number} refused: ${l.refused}`, more: [`at ${xy(l.cup_mm)}, ${cm(l.cup_mm)}`]}
      : {chip: "target", cls: "cup", head: `Target: cup ${l.number} of ${l.of} at ${xy(l.cup_mm)}, ${cm(l.cup_mm)}`, more: ["remembered in world mm; the arm won't look at it again before the drop"]};
    case "closeup": {
      const r = l.readings || [];
      return {chip: "measure", cls: "ball", head: `Close-up: ${r.length} readings agree within ${l.spread_mm} mm`,
        more: [`median ${xy(l.median_mm)}; the (−15, +20) camera correction is added to get the grab point`, ...(l.spread_mm > 6 ? ["readings spread > 6 mm: arm or ball still moving, median used anyway"] : [])]};
    }
    case "grab": {
      const o = l.outcome, p = l.finger_pos;
      const m = {"closed on something": ["ok", `Fingers stopped at ${p}: something is between them`, "lifting slowly to z 150, then checking it's still held"],
                 "closed on nothing": ["bad", `Fingers closed fully (${p} ≤ 50): missed the ball`, "open, lift, put the camera back over the spot, look again"],
                 "blocked": ["bad", `Fingers stuck at ${p} (≥ 800): landed on something, not around the ball`, "open, back off to z 150, look again"]}[o] || ["", `Gripper: ${o}`, ""];
      return {chip: "grab", cls: m[0], head: m[1], more: [`closed at ${xy(l.target_mm)}, z ${l.target_mm ? Math.round(l.target_mm[2]) : "?"}`, m[2]].filter(Boolean)};
    }
    case "result": {
      const more = [];
      if (l.failure) more.push(`reason: ${l.failure.replaceAll("_", " ")}`);
      if (l.miss_mm) more.push(`landed ${(l.miss_mm.x / 10).toFixed(0)} cm past, ${(l.miss_mm.y / 10).toFixed(0)} cm left of the cup`);
      return {chip: "result", cls: l.hit ? "ok" : "bad", head: l.hit ? "HIT: the ball is in the cup" : "No hit this attempt", more};
    }
    case "session": return {chip: "fill", cls: "ok", head: `Fill session: ${l.filled} cup(s) filled in ${l.attempts} attempt(s)`, more: [`cups left: ${l.left || "none"}`]};
  }
  return {chip: l.event || "?", head: l.event || "log line", more: [JSON.stringify(l).slice(0, 200)]};
}

function plan(lines) {
  const last = lines[lines.length - 1];
  if (!last) return ["Waiting for main.py to start an attempt.", "Each attempt makes a new folder in observations/detections/; this page switches to it by itself."];
  const target = [...lines].reverse().find(l => l.event === "target" && !l.refused);
  const tgt = target ? `cup ${target.number} at ${xy(target.cup_mm)}` : "the target cup";
  const st = last.stage || "";
  if (st.startsWith("cup_view")) {
    const i = +st.replace("cup_view_", "");
    return [`Surveying the table with an empty gripper, from view ${i} of 7 (${viewName(last.view)}). Every frame runs two cup detectors: the pink rim colour and YOLO.`,
            i < 7 ? `view ${i + 1} next (views it can't reach are skipped), then group all sightings into cups.` : "merge all sightings: within 5 cm = the same cup, keep cups seen ≥ 3 times, drop cups beyond 95 cm."];
  }
  if (["survey", "discarded", "yolo_vs_rim"].includes(last.event)) return ["Survey finished. Every cup is now remembered in world coordinates.", "choose the nearest cup within reach as the target."];
  if (last.event === "target") return last.refused ? [`The chosen cup can't be used (${last.refused}).`, "this attempt ends without picking a ball."]
    : [`Target locked: ${tgt}.`, "go to the look pose (350, 0, 445) and find a ball, ignoring balls within 9 cm of a cup or too close to the cups."];
  if (isBall(last)) {
    const ch = (last.items || []).find(b => b.chosen);
    if (st.startsWith("ball_scan")) return [`Scanning the table for every ball, view ${st.replace("ball_scan_", "")} (scripts/scan_table.py, read-only).`, "the next scan view."];
    if (st.startsWith("pick_retry")) return ["Looking again at the spot where the grab missed.", ch ? "the ball is still there: grab again at the new position." : "if the ball is gone or moved over 10 cm, give up on it."];
    if (last.closeup) return ["Camera is right above the ball, taking 3 close-up frames (the median is used).", "if coarse and close-up differ by more than 4 cm, start over (bumped, or not a ball)."];
    if (ch) return [`Ball spotted at ${xy(ch.world_mm)} from far away (accurate to a few cm).`, "move the camera straight above it (gripper z 160) and measure again from close."];
    if (st === "find_ball") return ["No usable ball from the look pose.", "search the table: the top view from z 800 first, then 6 side views, stopping at the first ball."];
    return ["Nothing usable in this search view.", "the next search view; if all are empty, the attempt ends with no ball found."];
  }
  if (last.event === "closeup") {
    if (st.startsWith("pick_retry")) return ["The ball is still there after the miss.", "grab again: hover at z 150, slow down, close at z 70."];
    return [`Exact ball position measured (${xy(last.median_mm)}, before the camera correction).`, "open, hover at z 150, slow to 15 °/s, go down to z 70, wait 0.5 s, close the fingers."];
  }
  if (last.event === "grab") return last.outcome === "closed on something"
    ? ["The ball is in the fingers (probably).", `lift slowly and check the fingers again, then carry it to ${tgt} and open 19 cm above the rim. If the move can't be planned, the ball is put back.`]
    : ["The grab missed.", "open, lift, look at the spot again and retry (up to 3 times)."];
  if (last.event === "result" || st === "result") return ["This attempt is over.", "wait for the next attempt (a new log folder)."];
  return ["Working…", ""];
}

function render() {
  const L = S.lines;
  // steps
  let cur = -1;
  for (const l of L) { const p = phaseOf(l); if (p) cur = Math.max(cur, PHASES.findIndex(x => x[0] === p)); }
  const lastP = L.length ? phaseOf(L[L.length - 1]) : null;
  const now = lastP ? PHASES.findIndex(x => x[0] === lastP) : cur;
  $("steps").innerHTML = PHASES.map(([k, t, s], i) =>
    `<li class="${i === now ? "now" : i < cur || (i < now) ? "done" : ""}">${t}<small>${s}</small></li>`).join("");
  // plan
  const [a, b] = plan(L);
  $("plan").innerHTML = `<div class="now">${esc(a)}</div>` + (b ? `<div class="next"><b>Next:</b> ${esc(b)}</div>` : "");
  // feed (newest first), with reading numbers per cup view
  const counts = {}, entries = L.map((l, i) => {
    if (isCup(l)) counts[l.stage] = (counts[l.stage] || 0) + 1;
    return {i, l, d: describe(l, counts[l.stage] || 1)};
  }).reverse();
  const sel = S.pinned ?? lastImageIndex();
  $("feed").innerHTML = entries.slice(0, 150).map(({i, l, d}) => `
    <li ${l.annotated ? `data-img="${i}" tabindex="0"` : ""} class="${i === sel ? "sel" : ""}">
      <div class="top"><time>${clock(l.t)}</time><span class="chip ${d.cls || ""}">${esc(d.chip)}</span>
        ${l.annotated ? '<span class="cam">camera frame, click to view</span>' : ""}</div>
      <div class="head">${esc(d.head)}</div>
      ${d.sub ? `<div class="cam">${esc(d.sub)}</div>` : ""}
      ${(d.more || []).length || (d.rej || []).length ? `<ul>${(d.more || []).map(m => `<li>${esc(m)}</li>`).join("")}${(d.rej || []).map(r => `<li class="rej">rejected ${esc(r)}</li>`).join("")}</ul>` : ""}
    </li>`).join("") || '<li>No decisions logged yet.</li>';
  renderFrame();
}

function lastImageIndex() {
  for (let i = S.lines.length - 1; i >= 0; i--) if (S.lines[i].annotated) return i;
  return null;
}

function renderFrame() {
  const i = S.pinned ?? lastImageIndex();
  $("live").hidden = S.pinned == null;
  if (i == null) {
    $("frame").innerHTML = '<div class="empty">Waiting for the first camera frame…</div>';
    $("caption").innerHTML = ""; $("found").innerHTML = ""; $("strip").innerHTML = ""; return;
  }
  const l = S.lines[i], name = $("drawn").checked ? l.annotated : l.image;
  const src = img(name);
  const el = $("frame").querySelector("img");
  if (!el || el.dataset.src !== src) $("frame").innerHTML = `<img alt="Camera frame ${esc(l.stage)}" data-src="${esc(src)}" src="${esc(src)}">` + (S.pinned != null ? '<span class="pin">older frame</span>' : "");
  const d = describe(l, 1);
  $("caption").innerHTML = `<b>${esc(d.head)}</b><span class="meta">${esc(l.stage)} · ${esc(l.detector)} · ${clock(l.t)}${l.view ? " · " + esc(viewName(l.view)) : ""}</span>`;
  const items = l.items || [];
  $("found").innerHTML = items.map(it => {
    const col = it.kept === false ? "#969696" : {rim: "var(--rim)", yolo: "var(--yolo)", ball: "var(--ball)", red_body: "#3d8bff"}[it.kind] || "var(--ink2)";
    const txt = it.kept === false ? `${it.kind} rejected: ${it.why || "?"}`
      : `${it.kind}${it.chosen ? " (chosen)" : ""}${it.duplicate_of ? ` = same cup as ${it.duplicate_of}` : ""}${it.world_mm ? ` at ${xy(it.world_mm)}, ${cm(it.world_mm)}` : ""}${it.confidence != null ? `, confidence ${it.confidence}` : ""}`;
    return `<div class="${it.kept === false ? "no" : ""}"><i style="background:${col}"></i><span>${esc(txt)}</span></div>`;
  }).join("") || '<div class="no"><i style="background:#969696"></i><span>Nothing detected in this frame.</span></div>';
  const shots = S.lines.map((l, j) => [l, j]).filter(([l]) => l.annotated).slice(-14).reverse();
  $("strip").innerHTML = shots.map(([l, j]) => `<button data-img="${j}" class="${j === i ? "on" : ""}" aria-label="Frame ${esc(l.stage)}"><img loading="lazy" alt="" src="${esc(img(l.annotated))}"><span>${esc(l.stage)}</span></button>`).join("");
}

async function poll() {
  try {
    const q = new URLSearchParams({folder: S.pick, have: S.folder || "", from: S.lines.length});
    const r = await (await fetch("/api/state?" + q)).json();
    if (r.folder !== S.folder || r.start === 0 && S.lines.length) { S.folder = r.folder; S.lines = []; S.pinned = null; }
    if (r.lines.length) { S.lines.push(...r.lines); S.lastNew = Date.now(); render(); }
    else if (!S.lines.length) render();
    const opts = [`<option value="">Latest attempt (live)</option>`, ...r.folders.map(f => `<option value="${esc(f)}">${esc(f)}</option>`)].join("");
    if ($("folder").dataset.opts !== opts) { $("folder").innerHTML = opts; $("folder").dataset.opts = opts; }
    $("folder").value = S.pick;
    const last = S.lines.length ? S.lines[S.lines.length - 1].t * 1000 : 0, age = (Date.now() - last) / 1000;
    const active = !S.pick && last && age < 20;
    $("dot").className = "dot " + (active ? "live" : r.folder ? "idle" : "");
    $("status").textContent = !r.folder ? "no attempts logged yet" : S.pick ? `replaying ${r.folder}`
      : active ? `live · ${r.folder}` : `${r.folder} · last decision ${age < 3600 ? Math.round(age) + " s" : Math.round(age / 3600) + " h"} ago`;
    const n = r.narration;
    $("consoleCard").hidden = !n; $("consoleHint").hidden = !!n;
    if (n) {
      const box = $("console"), stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 20;
      box.textContent = n.lines.join("\n");
      if (stick) box.scrollTop = box.scrollHeight;
    }
  } catch (e) {
    $("dot").className = "dot"; $("status").textContent = "can't reach live_view.py (is it still running? is the SSH tunnel up?)";
  }
}

function pinFrom(ev) {
  const t = ev.target.closest("[data-img]"); if (!t) return;
  S.pinned = +t.dataset.img === lastImageIndex() ? null : +t.dataset.img; render();
}
$("feed").addEventListener("click", pinFrom);
$("feed").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pinFrom(e); } });
$("strip").addEventListener("click", pinFrom);
$("live").addEventListener("click", () => { S.pinned = null; render(); });
$("drawn").addEventListener("change", renderFrame);
$("folder").addEventListener("change", e => { S.pick = e.target.value; S.folder = null; S.lines = []; S.pinned = null; poll(); });
poll(); setInterval(poll, 1000);
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = this machine only (reach it over an SSH tunnel)")
    ap.add_argument("--narration", type=Path, default=NARRATION,
                    help="file main.py's output is tee'd to (shown on the page when it exists)")
    a = ap.parse_args()
    Handler.narration_file = a.narration
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"live view on http://{a.host}:{a.port}  (reading {ROOT})")
    print(f"over SSH, on your laptop:  ssh -L {a.port}:localhost:{a.port} <user>@<this machine>"
          f"  then open http://localhost:{a.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
