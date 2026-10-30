# CLAUDE.md

Robot beer pong: a UFactory xArm 850 (Viam machine **sfhack11**, wrist RealSense `cam`,
2-finger `gripper`) finds a ping-pong ball (40 mm, orange or white), picks it, finds a red
cup, and gets the ball in. Phase 1 (pick → carry → **drop** over the cup) works. The current
goal is Phase 2: **throw** the ball into a cup out of drop reach.

## Safety rules (physical robot: these rules come first)
- Every script is a **dry run by default**. `--go` moves the arm. Don't run `--go`
  yourself unless the user explicitly asks, and remind them to have someone on the E-stop.
- Normal moves go through Viam motion planning (`motion.move` on `"gripper"`, world frame)
  so the table, wall and ceiling obstacles are checked. **Direct joint moves
  (`arm.move_to_joint_positions`) skip obstacle checks.** Throws need them for speed, so
  a throw script must check its own swept path (`throw_safety.path_problems`, used by `throw.plan`/`throw.throw`).
- Any speed or acceleration raised with `arm.do_command({"set_speed": ...})` must be
  restored in a `finally`.
- If something fails while holding the ball: put it back, open, go home
  (`main.put_ball_back`, `release_if_holding`). Keep this behavior in new flows.
- Never write Viam config changes or credentials. Keys live in `.env`
  (`.env.example`); `config/*.json` are read-only snapshots.

## Conventions
- Units: **mm, world frame**, arm base = (0,0,0). +x = away from the robot, toward the
  cups. Table top z ≈ -23. Orientation is Viam orientation vector `o_x,o_y,o_z,theta`.
- Joint indices are Viam 0–5: 0 base yaw, 1 shoulder, 2 elbow, 3 forearm roll,
  4 wrist pitch, 5 wrist roll (degrees).
- `THETA = 180` for every move (camera image upright). The camera calibration
  (`pick_ball.CAM_CORRECTION_MM`, `cup_aim.*_CORRECTION_MM`) was measured at that wrist
  angle. Changing theta flips the offsets.
- `"gripper"` frame = flange + 105 mm. Ball center ≈ flange + 175 mm when held.
- Home: `x=350 y=0 z=445`, pointing down (`o_z=-1`), theta 180.
- Obstacles (LIVE config, read Oct 29 21:50): table top -23, wall-back face x=-525, wall-left
  face y=625, ceiling bottom z=1200 (box center 1250). `config/sfhack11.standalone.json` still
  says 1000; the live machine wins. `throw_safety.py` uses the live values.
- Constants are module-level CAPS with a comment saying how they were measured
  ("worked on sfhack11 Oct 29 ..."). Keep that habit when tuning.
- Code style: small async functions, `go` flag passed down, `print` narration of each
  step. No classes unless needed. Match it.

## Layout
- `main.py`: full loop: home → `release_if_holding` → `survey_cups` (gripper empty, every
  `CUP_SCAN` view, 3 frames each, `merge_sightings` → distinct cups nearest-first, all
  **remembered in world frame**) → `find_ball` → `pick` → `drop_in_cup` at the chosen cup
  (no re-look) → home → ask hit/miss → `log_attempt`. `--ball orange|white|any`,
  `--autotune`, `--runs N` (series + summary), `--cup N` (default: nearest reachable).
  Cups farther than `drop.REACH_MAX_MM` (950) are DISCARDED right after the survey (location
  printed, `discarded` event) and never targeted.
  `--fill-cups` (`fill_all_cups`): cups stay put, several balls near the robot. Survey ONCE,
  then one ball per reachable cup, nearest first; a hit discards the cup (location printed,
  `discarded` event in the survey's log folder), a miss retries it (`MAX_TRIES_PER_CUP`).
  The survey sets `pick_ball.AVOID_XY`: balls within 9 cm of a cup are never picked.
  `cup_aim.find_cups` returns every cup in the current view; `find_cup` = first of those.
- `pick_ball.py`: look, center the camera over the ball, re-detect, then descend / close /
  lift at `PICK_SPEED_DEG_S` (`slow()` restores 60 deg/s in a `finally`). Grab at `GRAB_Z=70`.
  Missed grab → `look_again` (open, lift, camera over the spot, re-measure) and retry, up
  to `PICK_RETRIES` (3); gives up if the ball is gone or > 10 cm away, ending open and up.
  `ball_finder.py` / `white_ball.py`: OpenCV ball detection (orange HSV, white via depth).
- `cup_aim.py`: angled scan views (`CUP_SCAN`). `find_cups` runs, per frame, the rim method
  (`cup_rim.py`: pink-white opening + red band; positions calibrated by real drops) AND YOLO
  (`cup_detection.py`), and `combine`s them: same cup (rim box center inside YOLO box) →
  rim position + YOLO-vs-rim offset recorded; YOLO-only cups → YOLO depth position +
  `cup_detection.CUP_CORRECTION_MM` = (-54, +22), measured from 163 shared sightings;
  re-check with each survey's "YOLO vs rim" line). Survey needs ≥ 3 sightings per cup. Viam's red `cup-detector` only when neither found anything. `--no-yolo` = rim only.
- `cup_detection.py`: YOLO11m-seg (COCO "cup", `cup_vision/core.py`) + `screen()`: keeps only
  red solo cups (pink opening ≥ 18% of the mask; paper cups / cans ≤ 9% on real frames),
  drops cut-off openings and boxes inside stronger ones. 3D: rim ellipse fit, else a circle
  fit to the top 15 mm of depth points. `rotation_search` is off (only added duplicates).
  `cup_vision/`: the same detector as a Viam vision service (Rohan).
- `drop.py`: release pose over the rim. Straight down to x≈650. Beyond that, tilt
  forward `o=(0.7,0,-0.7)`. Reach limit ≈ 95 cm.
- `throw.py`: two presets. `far` (default): open-hand toss, cups ~80-100 cm, model matched
  the real landing within ~8 cm. `power`: full-force, direct mode, 1-2 m, timed release
  (scatters more). `plan()` = speed from the preset table x the Atlas correction, release time,
  joint limits + swept-path check, no robot. `throw()` runs a plan from home. Requires
  `--i-checked-with-staff`. Whip / pitch / lob / the first draft swing were cut Oct 29
  (0 hits; still in git history).
- `throw_safety.py`: URDF forward kinematics + `path_problems` (from Rohan's toss_ball.py).
- `throw_fit.py`: `correction(preset)` = median(observed / model landing) over recent Atlas
  throws (≥3, clamped 0.8-1.25). Docs with `exclude_from_fit` or bogus distances are skipped.
- `pick_and_throw.py`: home → (`--aim-cup`) scan + aim → plan + check BEFORE picking → pick →
  throw → ask outcome (`in` / `rim` / ±cm long[,±cm left] / `drop`) → log `phase: throw`
  with `throw.preset`, `cup`, `result.outcome`.
- `logger.py`: MongoDB Atlas `balltoss.attempts` (no `MONGODB_URI` → readable
  `attempt.json` in the detection-log folder, never dumped to the terminal); `suggest_offset()` = autotune from
  recent `miss_mm`. `telemetry.py`: per-move timings + snapshots in the attempt doc.
  `dashboard.py`, `stats.py`: hit-rate views.
- `detection_log.py`: per attempt, `observations/detections/<time>/` gets every frame the
  cup/ball detectors saw (raw + `_det.jpg` with candidates drawn) and `log.jsonl` (stage,
  view pose, every candidate kept/rejected with reason and world mm, survey/target/result
  events). Detectors call `detection_log.record(...)`; no-op unless `main.attempt` started a
  log. `python detection_log.py` summarizes the newest folder.
- `scripts/gripper_timing_test.py`, `scene_detection.py`: diagnostic / white-ball helpers.
- Docs: `TASKS.md` (owners, calibration history, log schema incl. the Phase 2 `throw`
  field), `DEMO.md`, `DASHBOARD.md`, `docs/cup-detection.md`.

## Throw: what we learned on Oct 29 (73 logged throws, 0 hits)
- FAR: the 3 throws with a real long/short measurement landed within ~8 cm of the model.
  Every Oct 29 `long_cm: 0` means "not measured" (they're flagged `exclude_from_fit`); new runs
  log an explicit outcome.
- `gripper.open()` takes ~0.85-1.15 s to return; the swing is 0.36-0.8 s. Timed releases
  (whip/power/pitch) land at a random point in the swing → big scatter. FAR works because the
  ball sits in the palm and leaves when the arm brakes.
- Fixes not in code yet: water in the cups (bounce-outs), a cup triangle (bigger target),
  a spoon/cradle in the gripper (no timing at all), running the throw on the robot computer
  as a Viam module (no 15-400 ms network jitter).
- Arm config has `collision_sensitivity: 0` (xArm collision detection off). Ask staff.

## Running
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate "$PWD/.conda"
python check_connection.py      # read-only
python main.py                  # dry run of the full loop
python -m pytest -q tests       # offline, no robot (needs pytest + mongomock)
```
Dry runs still connect to the robot (they read the camera). Tests don't.

## Git
Team repo `gvrohan11/Ball-Toss-In-Cup`. Work on a branch and open a PR into `main`.
Never commit `.env`, `observations/`, or model files.
