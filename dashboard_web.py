"""Live demo dashboard: reads MongoDB Atlas every few seconds and serves one page.

    python dashboard_web.py                 # http://localhost:8765 (opens the browser)
    python dashboard_web.py --hours 4       # only the last 4 h of attempts (default 10)
    python dashboard_web.py --port 9000

Run it on the LAPTOP (needs MONGODB_URI in .env and pymongo, like logger.py) and put the page on
the second screen. Nothing here talks to the robot. Read-only on Atlas.

What it shows (all from balltoss.attempts):
  - headline: throws in / thrown, drop hits, the learned correction (throw_fit)
  - cup map (top-down, robot at the bottom): the latest survey's cups, filled ones lit,
    a line from the base for every throw (green = in, amber = rim, gray = miss)
  - "the robot learns": swing speed vs where the ball landed - model (dashed) vs learned
    (model x correction, orange) and every measured throw as a dot
  - feed of the latest throws
"""
import argparse
import json
import math
import threading
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import throw
import throw_fit

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ball Toss Live</title>
<style>
:root{
  color-scheme:dark;
  --surface:#1a1a19; --card:#232321; --line:#34342f; --grid:#2c2c28;
  --ink:#f0efec; --ink-2:#c3c2b7; --ink-3:#8e8d86;
  --series:#d95926;            /* throws / learned (validated dark categorical slot) */
  --good:#0ca30c; --warn:#fab219; --miss:#8e8d86;
  --glow:rgba(217,89,38,.18);
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:22px 16px 32px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px;margin-bottom:18px}
h1{margin:0;font-size:26px;letter-spacing:.2px;overflow-wrap:anywhere}
h1 .pump{filter:saturate(1.2)}
.sub{color:var(--ink-2)}
.live{margin-left:auto;display:flex;align-items:center;gap:8px;color:var(--ink-2);font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 0 0 rgba(12,163,12,.6);animation:pulse 2s infinite}
.dot.off{background:#d03b3b;animation:none}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(12,163,12,.5)}70%{box-shadow:0 0 0 8px rgba(12,163,12,0)}100%{box-shadow:0 0 0 0 rgba(12,163,12,0)}}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:12px}
.tile,.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.tile .k{color:var(--ink-2);font-size:13px}
.tile .v{font-size:44px;font-weight:750;line-height:1.05;margin:6px 0 2px;font-variant-numeric:tabular-nums}
.tile .v small{font-size:22px;color:var(--ink-2);font-weight:600}
.tile .n{color:var(--ink-3);font-size:13px}
.tile.hero{background:linear-gradient(160deg,#2b211b,var(--card) 60%);border-color:#4a3122}
.grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:12px}
@media (max-width:900px){.grid{grid-template-columns:minmax(0,1fr)}}
.card h2{margin:0 0 2px;font-size:15px;font-weight:650}
.card .cap{color:var(--ink-3);font-size:13px;margin-bottom:8px}
svg{width:100%;height:auto;display:block}
.axis text,.lbl{fill:var(--ink-3);font-size:11px}
.axis line,.axis path{stroke:var(--grid)}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--ink-2);font-size:12px;margin-top:6px}
.legend i{display:inline-block;width:14px;height:0;border-top:2px solid var(--series);vertical-align:middle;margin-right:6px}
.legend i.model{border-top:2px dashed var(--ink-3)}
.legend b{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.feed{width:100%;border-collapse:collapse;font-size:13px}
.feed th{color:var(--ink-3);font-weight:500;text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
.feed td{padding:7px 8px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
.chip{display:inline-flex;align-items:center;gap:6px;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid}
.chip.in{color:var(--good);border-color:rgba(12,163,12,.5);background:rgba(12,163,12,.10)}
.chip.rim{color:var(--warn);border-color:rgba(250,178,25,.45);background:rgba(250,178,25,.08)}
.chip.miss{color:var(--ink-2);border-color:var(--line)}
.wide{grid-column:1/-1}
#tip{position:fixed;pointer-events:none;background:#0f0f0e;border:1px solid var(--line);border-radius:8px;padding:7px 10px;font-size:12px;color:var(--ink);display:none;z-index:9;max-width:240px}
.err{color:#ff8a8a;font-size:13px;margin-left:8px}
.muted{color:var(--ink-3)}
</style></head>
<body><main>
<header>
  <h1><span class="pump">🎃</span> Ball Toss <span class="muted" style="font-weight:500">— robot beer pong</span></h1>
  <span class="sub">UFactory xArm 850 · Viam · every attempt in MongoDB Atlas</span>
  <span class="live"><span id="dot" class="dot"></span><span id="upd">connecting…</span><span id="err" class="err"></span></span>
</header>

<section class="tiles">
  <div class="tile hero"><div class="k">Throws in the cup</div><div class="v" id="t-in">–</div><div class="n" id="t-in-n">&nbsp;</div></div>
  <div class="tile"><div class="k">Cups filled this run</div><div class="v" id="t-fill">–</div><div class="n" id="t-fill-n">&nbsp;</div></div>
  <div class="tile"><div class="k">Learned from its own throws</div><div class="v" id="t-fit">–</div><div class="n" id="t-fit-n">&nbsp;</div></div>
  <div class="tile"><div class="k">Drops in the cup</div><div class="v" id="t-drop">–</div><div class="n" id="t-drop-n">&nbsp;</div></div>
</section>

<section class="grid">
  <div class="card">
    <h2>Table, seen from above</h2>
    <div class="cap">Robot base at the bottom. Cups from the latest survey, numbered right → left from behind the robot.</div>
    <svg id="map" viewBox="0 0 640 470" role="img" aria-label="Top-down map of the cups and throws"></svg>
    <div class="legend">
      <span><b style="background:var(--good)"></b>✓ in</span>
      <span><b style="background:var(--warn)"></b>↩ rim (bounced out)</span>
      <span><b style="background:var(--miss)"></b>✗ miss</span>
    </div>
  </div>
  <div class="card">
    <h2>The robot learns how far it throws</h2>
    <div class="cap">Swing speed vs where the ball landed (POWER throw). Air drag makes balls fall short of the arm model; the robot measures that from Atlas and picks its speed from the learned line.</div>
    <svg id="fit" viewBox="0 0 560 360" role="img" aria-label="Landing distance against swing speed"></svg>
    <div class="legend">
      <span><i class="model"></i>arm model</span>
      <span><i></i>learned (model × correction)</span>
      <span><b style="background:var(--series)"></b>measured throw</span>
    </div>
  </div>
  <div class="card wide">
    <h2>Latest throws</h2>
    <div style="overflow-x:auto"><table class="feed"><thead><tr><th>Time</th><th>Result</th><th>Cup</th><th>Distance</th><th>Speed</th><th>Landed</th></tr></thead><tbody id="feed"></tbody></table></div>
  </div>
</section>
<div id="tip"></div>
</main>
<script>
const NS="http://www.w3.org/2000/svg";
const $=id=>document.getElementById(id);
function el(tag,attrs,parent,text){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(text!=null)e.textContent=text;if(parent)parent.appendChild(e);return e;}
const tip=$("tip");
function hover(node,html){node.addEventListener("mousemove",ev=>{tip.innerHTML=html;tip.style.display="block";tip.style.left=(ev.clientX+14)+"px";tip.style.top=(ev.clientY+14)+"px";});node.addEventListener("mouseleave",()=>tip.style.display="none");}
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COL={in:"--good",rim:"--warn",miss:"--miss"};
const ICON={in:"✓",rim:"↩",miss:"✗"};
function fmtTime(s){const d=new Date(s);return d.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});}

function drawMap(d){
  const svg=$("map");svg.innerHTML="";
  const W=640,H=470,bx=W/2,by=H-34;
  const cups=d.cups.length?d.cups:[];
  const all=cups.map(c=>c.pos).concat(d.throws.map(t=>t.cup).filter(Boolean));
  let maxF=700,maxS=300;all.forEach(p=>{maxF=Math.max(maxF,p[0]+110);maxS=Math.max(maxS,Math.abs(p[1])+110);});
  const s=Math.min((H-60)/maxF,(W/2-24)/maxS);   // px per mm, fit forward and sideways
  const maxR=Math.hypot(maxF,maxS);
  const P=p=>[bx-p[1]*s,by-p[0]*s];          // world x = forward (up), y = left
  // range rings
  [500,1000,1500,2000].filter(r=>r<maxR).forEach(r=>{
    el("circle",{cx:bx,cy:by,r:r*s,fill:"none",stroke:"#45453f","stroke-dasharray":"3 6"},svg);
    const a=Math.PI/180*58;el("text",{x:bx-r*s*Math.sin(a)-4,y:by-r*s*Math.cos(a),"text-anchor":"end",class:"lbl"},svg,(r/10)+" cm");
  });
  // throws (oldest first, so the newest is on top)
  d.throws.slice().reverse().forEach(t=>{
    if(!t.cup)return;const [x,y]=P(t.cup);
    const ln=el("line",{x1:bx,y1:by,x2:x,y2:y,stroke:css(COL[t.kind]),"stroke-width":t.kind==="in"?2.5:1.5,opacity:t.kind==="miss"?.45:.8,"stroke-linecap":"round"},svg);
  });
  // cups
  const R=Math.max(10,47*s);                 // solo cup opening ~95 mm
  cups.forEach(c=>{
    const [x,y]=P(c.pos);const k=c.state;
    const g=el("g",{},svg);
    if(k==="in")el("circle",{cx:x,cy:y,r:R+7,fill:"none",stroke:css("--good"),"stroke-width":2,opacity:.35},g);
    el("circle",{cx:x,cy:y,r:R,fill:k==="in"?css("--good"):"#3a1d17",stroke:k==="in"?css("--good"):"#c4462c","stroke-width":2.5},g);
    el("text",{x,y:y+4,"text-anchor":"middle","font-size":12,"font-weight":700,fill:k==="in"?"#0b1a0b":css("--ink")},g,k==="in"?"✓":c.n);
    el("text",{x,y:y-R-8,"text-anchor":"middle",class:"lbl"},g,c.side);
    el("circle",{cx:x,cy:y,r:R+8,fill:"transparent"},g);
    hover(g,`<b>cup ${c.n}</b><br>${(Math.hypot(c.pos[0],c.pos[1])/10).toFixed(0)} cm, ${c.side}<br>${k==="in"?"✓ filled":k==="rim"?"↩ hit the rim":"waiting"}`);
  });
  // robot
  el("rect",{x:bx-26,y:by-10,width:52,height:26,rx:7,fill:"#2f2f2b",stroke:css("--line")},svg);
  el("circle",{cx:bx,cy:by,r:9,fill:css("--series")},svg);
  el("text",{x:bx,y:by+30,"text-anchor":"middle",class:"lbl"},svg,"robot");
  if(!cups.length&&!d.throws.length)el("text",{x:W/2,y:H/2,"text-anchor":"middle",fill:css("--ink-3")},svg,"no throws yet");
}

function drawFit(d){
  const svg=$("fit");svg.innerHTML="";
  const W=560,H=360,m={l:52,r:16,t:14,b:40};
  const pts=d.points;
  const xs=d.model.map(p=>p[0]).concat(pts.map(p=>p.speed));
  let x0=Math.min(...xs),x1=Math.max(...xs);
  if(pts.length){x0=Math.max(40,Math.min(...pts.map(p=>p.speed))-12);x1=Math.min(180,Math.max(...pts.map(p=>p.speed))+12);}
  const inX=d.model.filter(p=>p[0]>=x0&&p[0]<=x1);
  const ys=inX.map(p=>p[1]).concat(inX.map(p=>p[1]*d.ratio)).concat(pts.map(p=>p.landed));
  let y0=Math.min(...ys)-40,y1=Math.max(...ys)+40;
  const X=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r), Y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
  const ax=el("g",{class:"axis"},svg);
  const step=(y1-y0)>600?200:(y1-y0)>250?100:50;
  for(let v=Math.ceil(y0/step)*step;v<=y1;v+=step){el("line",{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v)},ax);el("text",{x:m.l-8,y:Y(v)+4,"text-anchor":"end"},ax,(v/10)+" cm");}
  const xstep=(x1-x0)>60?20:10;
  for(let v=Math.ceil(x0/xstep)*xstep;v<=x1;v+=xstep){el("text",{x:X(v),y:H-m.b+18,"text-anchor":"middle"},ax,v+"°/s");}
  el("text",{x:(m.l+W-m.r)/2,y:H-6,"text-anchor":"middle",class:"lbl"},svg,"swing speed");
  const path=f=>inX.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+","+Y(f(p[1])).toFixed(1)).join("");
  el("path",{d:path(v=>v),fill:"none",stroke:css("--ink-3"),"stroke-width":2,"stroke-dasharray":"6 5"},svg);
  el("path",{d:path(v=>v*d.ratio),fill:"none",stroke:css("--series"),"stroke-width":2.5,"stroke-linecap":"round"},svg);
  pts.forEach(p=>{
    const g=el("g",{},svg);
    el("circle",{cx:X(p.speed),cy:Y(p.landed),r:6,fill:p.kind==="in"?css("--series"):css("--surface"),stroke:css("--series"),"stroke-width":2.5},g);
    el("circle",{cx:X(p.speed),cy:Y(p.landed),r:13,fill:"transparent"},g);
    hover(g,`${ICON[p.kind]} ${p.kind==="in"?"in the cup":p.kind==="rim"?"hit the rim":"miss"}<br>${p.speed.toFixed(0)}°/s → landed ${(p.landed/10).toFixed(0)} cm<br>model said ${(p.model/10).toFixed(0)} cm`);
  });
  // direct label on the learned line (right end)
  const last=inX[inX.length-1];
  if(last){el("text",{x:X(last[0])-4,y:Y(last[1]*d.ratio)+16,"text-anchor":"end",fill:css("--ink-2"),"font-size":12},svg,"learned ×"+d.ratio.toFixed(3));
           el("text",{x:X(last[0])-4,y:Y(last[1])-8,"text-anchor":"end",fill:css("--ink-3"),"font-size":12},svg,"arm model");}
  if(!pts.length)el("text",{x:W/2,y:H/2,"text-anchor":"middle",fill:css("--ink-3")},svg,"no measured throws yet");
}

function render(d){
  const thrown=d.throws.filter(t=>t.kind!=="skip");
  const ins=thrown.filter(t=>t.kind==="in").length;
  $("t-in").innerHTML=`${ins}<small> / ${thrown.length}</small>`;
  const run=thrown.filter(t=>t.run), runIn=run.filter(t=>t.kind==="in").length;
  $("t-in-n").textContent=thrown.length?(run.length?`this run ${runIn} of ${run.length} · `:"")+`${Math.round(100*ins/thrown.length)}% over ${d.hours} h incl. tuning`:"no throws yet";
  const filled=d.cups.filter(c=>c.state==="in").length;
  $("t-fill").innerHTML=d.cups.length?`${filled}<small> / ${d.cups.length}</small>`:"–";
  $("t-fill-n").textContent=d.cups.length?"cups in the latest survey":"no survey logged yet";
  $("t-fit").innerHTML=`×${d.ratio.toFixed(3)}`;
  $("t-fit-n").textContent=`ball flies ${Math.round(d.ratio*100)}% of the arm model · ${d.points.length} measured throws`;
  $("t-drop").innerHTML=`${d.drops.hits}<small> / ${d.drops.total}</small>`;
  $("t-drop-n").textContent=d.drops.total?"pick → carry → drop · last "+d.hours+" h":"none in this window";
  drawMap(d);drawFit(d);
  $("feed").innerHTML=d.throws.slice(0,12).map(t=>`<tr><td>${fmtTime(t.ts)}</td>
    <td><span class="chip ${t.kind}">${ICON[t.kind]||"–"} ${t.label}</span></td>
    <td>${t.cup_n?("cup "+t.cup_n):"–"}${t.side?(" · "+t.side):""}</td>
    <td>${t.dist?((t.dist/10).toFixed(0)+" cm"):"–"}</td>
    <td>${t.speed?t.speed.toFixed(0)+"°/s":"–"}</td>
    <td>${t.landed?((t.landed/10).toFixed(0)+" cm"):"–"}</td></tr>`).join("")||`<tr><td colspan="6" class="muted">no throws yet</td></tr>`;
}

async function tick(){
  try{
    const r=await fetch("/data.json",{cache:"no-store"});const d=await r.json();
    if(d.error)throw new Error(d.error);
    render(d);$("dot").className="dot";$("upd").textContent="live · updated "+new Date().toLocaleTimeString();$("err").textContent="";
  }catch(e){$("dot").className="dot off";$("err").textContent="Atlas: "+e.message;}
}
tick();setInterval(tick,3000);
</script></body></html>"""


def _kind(res):
    if not res:
        return "skip", "not recorded"
    if res.get("hit"):
        return "in", "in" + (f" cup {res['cup_number']}" if res.get("cup_number") else "")
    o = res.get("outcome")
    if o == "rim_out":
        return "rim", "rim"
    if o in ("not_recorded", "reached_cup_unclear") or res.get("hit") is None:
        return "skip", "not recorded"
    if o == "dropped":
        return "miss", "dropped"
    lc = res.get("long_cm")
    return "miss", (f"{abs(lc):.0f} cm {'long' if lc > 0 else 'short'}" if lc else "miss")


def _side(x, y):
    a = math.degrees(math.atan2(y, x))
    return "straight" if abs(a) < 3 else f"{abs(a):.0f}° {'left' if a > 0 else 'right'}"


def collect(col, hours):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    throws = list(col.find({"phase": "throw", "timestamp": {"$gte": since}}, {"_id": 0, "telemetry": 0})
                  .sort("timestamp", -1).limit(200))
    drops_q = {"phase": "drop", "timestamp": {"$gte": since}, "result.hit": {"$in": [True, False]}}
    drops = {"total": col.count_documents(drops_q), "hits": col.count_documents({**drops_q, "result.hit": True})}

    out = []
    for d in throws:
        t, res, cup = d.get("throw") or {}, d.get("result") or {}, d.get("cup") or {}
        if t.get("released") is False or t.get("check_only"):
            continue
        kind, label = _kind(res)
        pos = cup.get("pos_mm")
        out.append(dict(ts=d["timestamp"].isoformat() if hasattr(d.get("timestamp"), "isoformat") else str(d.get("timestamp")),
                        kind=kind, label=label, cup=pos[:2] if pos else None, cup_n=cup.get("number"),
                        side=_side(*pos[:2]) if pos else None, dist=cup.get("dist_mm") or t.get("distance_mm"),
                        speed=t.get("speed_dps", t.get("throw_speed")), landed=throw_fit.landing_mm(d),
                        hit_cup=res.get("cup_number"), survey=cup.get("survey") or [], ts_raw=d.get("timestamp"),
                        run=False))

    # cups: the latest logged survey; state from throws made with that same survey
    cups = []
    latest = next((o for o in out if o["survey"]), None)
    if latest:
        layout = latest["survey"]
        cups = [dict(n=i + 1, pos=p, side=_side(*p), state="open") for i, p in enumerate(layout)]
        for o in out:
            if o["survey"] != layout:
                continue
            o["run"] = o["kind"] != "skip"
            n = o["hit_cup"] or o["cup_n"]
            if n and 1 <= n <= len(cups):
                if o["kind"] == "in":
                    cups[n - 1]["state"] = "in"
                elif o["kind"] == "rim" and cups[n - 1]["state"] == "open":
                    cups[n - 1]["state"] = "rim"

    preset = "power"
    ratio = throw_fit.fit(throws, preset)
    points = []
    for r, d in throw_fit.ratios(throws, preset):
        spd = throw_fit.speed_of(d)
        kind, _ = _kind(d.get("result"))
        points.append(dict(speed=spd, landed=throw_fit.landing_mm(d), model=throw.model_landing(spd, preset),
                           kind=kind if kind != "skip" else "miss"))
    model = [[s, round(throw.model_landing(s, preset))] for s in range(50, 181, 2)]
    for o in out:
        o.pop("survey", None), o.pop("ts_raw", None)
    return dict(hours=hours, throws=[o for o in out], cups=cups, ratio=ratio, points=points, model=model,
                drops=drops, preset=preset)


def make_handler(col, hours):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/data.json"):
                try:
                    body = json.dumps(collect(col(), hours), default=str)
                except Exception as e:  # noqa: BLE001 - show it on the page, keep serving
                    body = json.dumps({"error": f"{type(e).__name__}: {str(e)[:120]}"})
                self._send(200, body, "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hours", type=float, default=10, help="only attempts from the last N hours")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    a = ap.parse_args()
    import logger

    def col():
        c = logger._get_collection()
        if c is None:
            raise RuntimeError("no MONGODB_URI in .env (or Atlas unreachable)")
        return c
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(col, a.hours))
    url = f"http://localhost:{a.port}"
    print(f"Ball Toss dashboard on {url}  (Ctrl+C to stop)")
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
