# Who does what

> Oct 29 night: `balls.py` / `cups.py` were removed (never used; detection lives in
> `pick_ball.py` / `ball_finder.py` and `cup_aim.py` / `cup_rim.py` / `cup_detection.py`).
> Throw status: see CLAUDE.md "Throw: what we learned".

**Phase 1:** pick the ball → move over the cup → drop it in.
**Phase 2 (later):** throw.

| Who | Owns | File and function |
|---|---|---|
| Manan | Integration, pick, drop, demo | `main.py`, `pick_ball.py`, `drop.py` |
| Ball teammate | Ball detection | `balls.py` → `find_ball(robot)` |
| Rohan | Cup detection | `cups.py` → `get_cups(robot)` |
| MongoDB teammate | Logging and dashboard | `logger.py` → `log_attempt(doc)` |

## Update (Thu ~2 PM): full loop works, 4/4 drops in the cup
- `python main.py --go`: home → find + pick ball → 30°/45° look → find cup → drop → home.
- Pick: center the camera over the ball, correction (-15, +20) mm at wrist θ=180, grab at z=70, wait for fingers.
- Drop: gripper tilted forward (o = 0.7, 0, -0.7), released ~19 cm above the rim. Reaches cups ~70–95 cm out.
- Home pose: x=350 y=0 z=445, pointing down, θ=180 (camera image upright).
- Ceiling obstacle raised to 1250 mm on sfhack11 (watch the camera cable).
- **Rohan:** the angled views (`cup_aim.CUP_LOOKS`) show the red sides. Gap: with the ball held,
  the ball can hide the red side → cup-detector misses. Needs rim detection or YOLO "cup".
- **Ball teammate:** ball-detector sometimes boxes only the glare spot; `pick_ball.best_ball` filters.

## Where we were (Thu ~1 PM)
- Machine **sfhack11** (xArm850, RealSense on the wrist, 2-finger gripper). Obstacles: table, ceiling, wall-back, wall-left.
- **Look pose** (camera straight down at the table): arm `x=350 y=0 z=550 o=(0,0,-1) θ=0`. Gripper tip is about 105 mm lower.
- Vision services in the Viam config:
  - `ball-detector`: 2D color detector for the orange ball. Works: one clean box.
  - `ball-3d`: 3D position of the ball from the depth camera. Works.
  - `cup-detector`: 2D red detector. Works on a cup lying on its side; **fails on an upright cup seen from above** (only the white inside is visible).
  - `cup-3d`: 3D version of cup-detector.
- Code on `main`: `connection.py`, `check_connection.py`, `pick_ball.py` (not yet run on the arm).

## Shared rules
- Positions are **mm in the `world` frame** `(x, y, z)`. The arm base is (0, 0, 0). The table top is about z = -23.
- Move with **motion planning** (`motion.move` on `"gripper"`). Direct joint moves ignore obstacles.
- **Say it out loud before saving a config change** in the Viam app. The ball and cup people both edit vision settings.
- **Someone on the E-stop** whenever the arm moves.
- Use your own branch and open a PR into `main`.
- Credentials go in `.env` (copy `.env.example`), never in git.

## Manan: integration
1. `python check_connection.py`, then `python pick_ball.py` (dry run), then `--go`.
2. Make `pick(robot, ball, go)` reliable (3 good picks in a row).
3. `drop.py`: move above the cup and open. Tune height and offset.
4. `main.py`: find ball → pick → find cups → drop → ask "hit? y/n" → `log_attempt`.
5. Own the Friday demo.

## Ball teammate: `balls.py` → `find_ball(robot)`
- Returns `(x, y, z)` in mm (world frame) of the ball to pick, or `None`.
- Start from `find_ball` in `pick_ball.py` (uses `ball-3d`), then move it into `balls.py`.
- Make it reliable:
  - Different spots on the table and different lighting.
  - Ignore anything outside the reachable table area.
  - Pick the right ball when there are several.
- White balls: the color detector cannot see white. Use YOLO ("sports ball" class) or stick to orange.
- Check: put the ball somewhere, print `find_ball()`, and measure with a ruler from the arm base.

## Rohan: `cups.py` → `get_cups(robot)`
- Returns a list of cup `(x, y, z)` in mm (world frame). The point we want is the **center of the cup opening (rim)**.
- Problem: from straight above, an upright cup shows white, not red.
- Options:
  1. An **angled look pose** that sees the red sides. Save it with the arm position saver.
  2. YOLO "cup" class.
  3. Depth: cups are tall (~120 mm) objects on the table.
- Then fix the z to the rim height, not the middle of the blob.
- Check: compare `get_cups()` with a ruler measurement.

## MongoDB teammate: `logger.py` → `log_attempt(doc)`
- Use the event Atlas cluster. `MONGODB_URI` goes in `.env` (add the key to `.env.example`). Use `pymongo`.
- Collection `balltoss.attempts`, one document per attempt:

```json
{
  "attempt_id": "uuid", "timestamp": "...", "session": "thu-evening-1", "phase": "drop",
  "ball":    {"color": "orange", "bbox_px": [538,500,615,587], "pos_mm": {"x":410,"y":-35,"z":-5}},
  "cup":     {"cup_id": 0, "pos_mm": {"x":520,"y":120,"z":90}},
  "pick":    {"grasp_pose": {"x":410,"y":-35,"z":0}, "holding": true, "duration_s": 6.2},
  "release": {"pose": {"x":520,"y":120,"z":240,"o_z":-1}, "height_mm": 150, "offset_mm": {"x":0,"y":0}},
  "result":  {"hit": false, "miss_mm": {"x":18,"y":-6}, "failure": "bounced_out"},
  "image_before": "...", "image_after": "...",
  "config_revision": "...", "notes": ""
}
```

- `failure` is one of: `no_ball_found`, `grasp_failed`, `dropped_in_transit`, `bounced_out`, `missed`.
- Phase 2 (`phase: "throw"`): `throw: {preset, distance_mm, speed_dps, release_delay_s, yaw_deg, direct, ratio, model_landing_mm, timing}`, `cup: {pos_mm, dist_mm}`, `result: {hit, outcome: in|rim_out|long|short|dropped, long_cm, left_cm}`. Oct 29 docs were back-filled with `throw.preset` (`preset_inferred: true`); junk ones have `exclude_from_fit: true`.
- Build:
  1. `log_attempt()`.
  2. Dashboard (Atlas Charts): hit rate over time, miss scatter around the cup, pick success, failure reasons.
  3. **Auto-tune:** average recent `miss_mm` → suggest the next `offset_mm`. This is the judges' favorite.
  4. Bonus: Vector Search on images.
- `main.py` builds the document. The MongoDB teammate owns storage and the dashboard. Log from attempt #1.

## Timeline
- **Thu, before the 9 PM close:** pick works, `find_ball` and `get_cups` return positions, `log_attempt` writes to Atlas.
- **Fri, by noon:** full loop works end to end, dashboard is live.
- **Fri 3:30 PM:** demo. Throw only if Phase 1 is solid.
