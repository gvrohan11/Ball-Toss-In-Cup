"""Build a one-page HTML dashboard from the attempts in MongoDB Atlas.

    python dashboard.py              # writes dashboard.html and opens it
    python dashboard.py --watch      # rebuilds every 10 s (page auto-refreshes)
    python dashboard.py --session thu-manual

Backup for Atlas Charts during the demo: no setup beyond MONGODB_URI in .env.
"""
import html
import json
import sys
import time
import webbrowser
from collections import Counter
from pathlib import Path

from logger import hit_rate, recent_attempts, suggest_offset

OUT = Path(__file__).resolve().parent / "dashboard.html"


def _svg_scatter(points, size=320, scale_mm=60):
    """Miss positions around the cup (center), robot's view: up = past the cup."""
    c = size / 2
    r_cup = 48 / scale_mm * c        # solo cup opening radius ~48 mm
    dots = "".join(
        f'<circle cx="{c - y / scale_mm * c:.1f}" cy="{c - x / scale_mm * c:.1f}" r="6" class="miss"/>'
        for x, y in points)
    return (f'<svg viewBox="0 0 {size} {size}" class="scatter" role="img" aria-label="Miss positions">'
            f'<line x1="{c}" y1="0" x2="{c}" y2="{size}" class="axis"/>'
            f'<line x1="0" y1="{c}" x2="{size}" y2="{c}" class="axis"/>'
            f'<circle cx="{c}" cy="{c}" r="{r_cup:.1f}" class="cup"/>{dots}'
            f'<text x="{c + 4}" y="14" class="lbl">past</text>'
            f'<text x="{c + 4}" y="{size - 6}" class="lbl">short</text>'
            f'<text x="4" y="{c - 4}" class="lbl">left</text>'
            f'<text x="{size - 34}" y="{c - 4}" class="lbl">right</text></svg>')


def build(session=None, refresh=None):
    hits, total = hit_rate(session)
    docs = recent_attempts(200, session)
    dx, dy = suggest_offset()
    rate = f"{100 * hits / total:.0f}%" if total else "–"
    failures = Counter(d.get("result", {}).get("failure") or "hit" for d in docs
                       if d.get("phase") != "test")
    misses = [(m["x"], m["y"]) for d in docs
              if (m := d.get("result", {}).get("miss_mm"))]
    durations = [d["duration_s"] for d in docs if isinstance(d.get("duration_s"), (int, float))]
    avg_dur = f"{sum(durations) / len(durations):.0f} s" if durations else "–"
    bars = "".join(
        f'<div class="bar"><span>{html.escape(str(k))}</span>'
        f'<div style="width:{100 * v / max(failures.values()):.0f}%"></div><b>{v}</b></div>'
        for k, v in failures.most_common())
    rows = "".join(
        f"<tr><td>{html.escape(str(d.get('timestamp', ''))[:19])}</td>"
        f"<td>{'✅' if d.get('result', {}).get('hit') else '❌' if d.get('result', {}).get('hit') is False else '–'}</td>"
        f"<td>{html.escape(str(d.get('result', {}).get('failure') or ''))}</td>"
        f"<td>{html.escape(json.dumps(d.get('cup', {}).get('pos_mm', '')))}</td>"
        f"<td>{html.escape(str(d.get('notes', '')))}</td></tr>"
        for d in docs[:15])
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    page = f"""<!doctype html><html><head><meta charset="utf-8">{meta}
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ball Toss Dashboard</title><style>
:root{{--bg:#0f1115;--card:#181b22;--fg:#e8eaf0;--mut:#9aa1ad;--acc:#ff7a2f;--ok:#3ecf8e;--line:#2a2f3a}}
body{{margin:0;font:15px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--mut);margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;align-items:start}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.big{{font-size:44px;font-weight:700;color:var(--acc)}} .lab{{color:var(--mut);font-size:13px}}
.scatter{{width:100%;max-width:320px}} .axis{{stroke:var(--line)}} .cup{{fill:none;stroke:var(--ok);stroke-width:2}}
.miss{{fill:var(--acc);opacity:.8}} .lbl{{fill:var(--mut);font-size:11px}}
.bar{{display:grid;grid-template-columns:140px 1fr 30px;gap:8px;align-items:center;margin:6px 0}}
.bar div{{height:10px;background:var(--acc);border-radius:5px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} td{{padding:6px;border-top:1px solid var(--line)}}
.wide{{grid-column:1/-1;overflow-x:auto}}
</style></head><body><main>
<h1>🤖🏓 Ball Toss — robot beer pong</h1>
<div class="sub">UFactory 850 + Viam · every attempt logged to MongoDB Atlas{' · session ' + html.escape(session) if session else ''}</div>
<div class="grid">
 <div class="card"><div class="lab">Hit rate</div><div class="big">{rate}</div><div class="lab">{hits} of {total} drops in the cup</div></div>
 <div class="card"><div class="lab">Attempts logged</div><div class="big">{len(docs)}</div><div class="lab">incl. {sum(v for k, v in failures.items() if k not in ("hit", "missed"))} stopped safely before the drop</div><div class="lab">avg {avg_dur} per attempt</div></div>
 <div class="card"><div class="lab">Auto-tune (from misses in Atlas)</div><div class="big">{dx:+.0f}/{dy:+.0f}</div><div class="lab">mm x / y applied with --autotune</div></div>
 <div class="card"><div class="lab">Where misses landed (cup = green ring)</div>{_svg_scatter(misses)}</div>
 <div class="card"><div class="lab">Outcomes</div>{bars or '<div class="lab">no data yet</div>'}</div>
 <div class="card wide"><div class="lab">Latest attempts</div><table>{rows}</table></div>
</div></main></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    session = sys.argv[sys.argv.index("--session") + 1] if "--session" in sys.argv else None
    if "--watch" in sys.argv:
        path = build(session, refresh=10)
        webbrowser.open(path.as_uri())
        while True:
            time.sleep(10)
            build(session, refresh=10)
    else:
        path = build(session)
        print("wrote", path)
        webbrowser.open(path.as_uri())
