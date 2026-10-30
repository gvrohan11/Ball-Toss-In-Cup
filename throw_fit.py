"""Learn the throw from Atlas: how far each preset REALLY throws vs its URDF model.

For every logged throw of a preset we know the swing speed and where the ball landed
(cup distance + how far long/short it was). correction(preset) = median(observed / model)
over the recent throws; throw.speed_for() scales the model table by it, so the next
throw's speed comes from the robot's own history.

    python throw_fit.py            # per-preset report from Atlas
    python throw_fit.py --far      # just FAR

Landing rules (per document):
  result.hit True, or outcome "in" / "rim_out"  -> landed at the cup distance
  result.long_cm != 0                           -> cup distance + long_cm
  legacy docs (no outcome) with long_cm 0       -> skipped (Oct 29: 0 was typed as "no measurement")
  outcome "dropped" / "wide_unknown", exclude_from_fit, distance outside 300-2500 mm -> skipped
"""
import statistics
import sys

MIN_THROWS = 3           # fewer than this -> no correction (1.0)
RECENT = 8               # newest N usable throws (Oct 30: 15 reacted too slowly to a 4-5 cm short bias)
CLAMP = (0.8, 1.25)      # never trust a correction outside this
DIST_OK = (300, 2500)    # mm; Oct 29 had hand-typed distances like 10 / 50 / 2000 as speed hacks


def landing_mm(doc):
    """Observed landing distance from the base (mm), or None if the doc can't tell us."""
    t, r = doc.get("throw") or {}, doc.get("result") or {}
    if doc.get("exclude_from_fit") or t.get("dry_run") or t.get("released") is False:
        return None
    d = t.get("distance_mm")
    if d is None or not DIST_OK[0] <= d <= DIST_OK[1]:
        return None
    outcome = r.get("outcome")
    if outcome in ("dropped", "wide_unknown"):
        return None
    if r.get("hit") is True or (outcome == "rim_out" and not r.get("long_cm")):
        return float(d)
    if outcome == "rim_out":                 # "rim front" / "rim back": short / long by ~a cup radius
        return d + 10.0 * r["long_cm"]
    long_cm = r.get("long_cm")
    if long_cm is None:
        return None
    if long_cm == 0 and outcome is None:
        return None              # legacy: 0 meant "not measured", not "right distance"
    return d + 10.0 * long_cm


def speed_of(doc):
    t = doc.get("throw") or {}
    return t.get("speed_dps", t.get("throw_speed"))


def ratios(docs, preset):
    """[(observed/model, doc)] for usable docs of this preset, newest first as given."""
    import throw
    out = []
    for doc in docs:
        t = doc.get("throw") or {}
        if t.get("preset") != preset or (t.get("scale") or 1.0) != 1.0:
            continue
        land, spd = landing_mm(doc), speed_of(doc)
        if land is None or spd is None:
            continue
        out.append((land / throw.model_landing(spd, preset), doc))
    return out


def fit(docs, preset):
    """correction factor from docs (newest first). 1.0 when there is too little data."""
    rs = [r for r, _ in ratios(docs, preset)][:RECENT]
    if len(rs) < MIN_THROWS:
        return 1.0
    return min(max(statistics.median(rs), CLAMP[0]), CLAMP[1])


def _docs(preset):
    import logger
    col = logger._get_collection()
    if col is None:
        return []
    return list(col.find({"phase": "throw", "throw.preset": preset}, {"_id": 0, "telemetry": 0})
                .sort("timestamp", -1).limit(200))


def correction(preset):
    """Live correction for a preset from Atlas; 1.0 if Atlas is unavailable or there's too little data."""
    try:
        c = fit(_docs(preset), preset)
    except Exception as e:  # noqa: BLE001 - never block a throw on the database
        print(f"   [fit] Atlas unavailable ({str(e)[:60]}); using the raw model")
        return 1.0
    print(f"   [fit] {preset}: learned correction x{c:.3f} from Atlas")
    return c


def report(preset):
    import throw
    docs = _docs(preset)
    rs = ratios(docs, preset)
    print(f"{preset.upper()}: {len(docs)} logged throws, {len(rs)} usable, correction x{fit(docs, preset):.3f}")
    for r, d in rs[:RECENT]:
        t = d["throw"]
        print(f"  {d.get('timestamp')}  {speed_of(d):5.0f} deg/s  cup {t['distance_mm']:5.0f}  "
              f"landed {landing_mm(d):5.0f}  model {throw.model_landing(speed_of(d), preset):5.0f}  x{r:.2f}")


if __name__ == "__main__":
    import throw
    for p in throw.PRESETS:
        if len(sys.argv) == 1 or f"--{p}" in sys.argv:
            report(p)
