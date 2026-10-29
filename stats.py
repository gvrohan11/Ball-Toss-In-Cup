"""Print what's in Atlas: hit rate, last attempts, suggested aim correction.

    python stats.py                 # all sessions
    python stats.py --session thu-1405
    python stats.py --seed-today    # one time: store the 5 hand-driven drops from Oct 29
"""
import sys
from datetime import datetime, timezone

from logger import hit_rate, log_attempt, recent_attempts, suggest_offset

# The five drops on Oct 29 that were driven move-by-move (all went in).
# Cup positions are the corrected rim centers used for each release.
TODAY = [
    ("13:39", (870, 20), "first drop, tilted reach to the far cup"),
    ("13:44", (870, 20), "repeat from the same pose"),
    ("13:49", (870, 28), "auto-centered on the cup; slightly off-center in y"),
    ("13:58", (728, -66), "cup moved closer (~78 cm), found with the 30 deg view"),
    ("14:05", (730, -66), "full run on video"),
]


def seed_today():
    for hhmm, (x, y), note in TODAY:
        h, m = map(int, hhmm.split(":"))
        ts = datetime(2026, 10, 29, h + 7, m, tzinfo=timezone.utc)  # PDT -> UTC
        log_attempt({"phase": "drop", "session": "thu-manual", "timestamp": ts,
                     "cup": {"pos_mm": [x, y, 110]}, "result": {"hit": True},
                     "notes": note, "driven": "move-by-move via Viam MCP"})


def main():
    if "--seed-today" in sys.argv:
        seed_today()
    session = sys.argv[sys.argv.index("--session") + 1] if "--session" in sys.argv else None
    hits, total = hit_rate(session)
    print(f"hit rate: {hits}/{total}" + (f" = {100 * hits / total:.0f}%" if total else ""))
    dx, dy = suggest_offset()
    print(f"suggested aim correction: {dx:+.0f} mm x, {dy:+.0f} mm y")
    print("last attempts:")
    for a in recent_attempts(10, session):
        r = a.get("result", {})
        print(f"  {a.get('timestamp')}  hit={r.get('hit')}  miss={r.get('miss_mm')}  "
              f"failure={r.get('failure')}  {a.get('notes', '')}")


if __name__ == "__main__":
    main()
