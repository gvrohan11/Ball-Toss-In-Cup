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
  a throw script must check its own swept path (see `toss_ball.path_problems`).
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
- Obstacles: table, wall-back (x ≈ -525), wall-left (y ≈ 625), ceiling. The ceiling's
  bottom is z=1000 in `config/sfhack11.standalone.json`. TASKS.md says it was raised to
  1250 on the live machine. Check the live config before relying on either.
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
  Cups farther than `drop.REACH_MAX_MM` (950) are refused BEFORE the pick.
  `cup_aim.find_cups` returns every cup in the current view; `find_cup` = first of those.
- `pick_ball.py`: look, center the camera over the ball, re-detect, then descend / close /
  lift at `PICK_SPEED_DEG_S` (`slow()` restores 60 deg/s in a `finally`). Grab at `GRAB_Z=70`.
  `ball_finder.py` / `white_ball.py`: OpenCV ball detection (orange HSV, white via depth).
- `cup_aim.py`: angled scan views (`CUP_SCAN`). `find_cups` runs, per frame, the rim method
  (`cup_rim.py`: pink-white opening + red band; positions calibrated by real drops) AND YOLO
  (`cup_detection.py`), and `combine`s them: same cup (rim box center inside YOLO box) →
  rim position + YOLO-vs-rim offset recorded; YOLO-only cups → YOLO depth position +
  `cup_detection.CUP_CORRECTION_MM` (still (0,0): set it from the survey's "YOLO vs rim"
  line). Viam's red `cup-detector` only when neither found anything. `--no-yolo` = rim only.
- `cup_detection.py`: YOLO11m-seg (COCO "cup", `cup_vision/core.py`) + `screen()`: keeps only
  red solo cups (pink opening ≥ 18% of the mask; paper cups / cans ≤ 9% on real frames),
  drops cut-off openings and boxes inside stronger ones. 3D: rim ellipse fit, else a circle
  fit to the top 15 mm of depth points. `rotation_search` is off (only added duplicates).
  `cup_vision/`: the same detector as a Viam vision service (Rohan).
  `cups.py` / `balls.py`: thin original teammate interfaces.
- `drop.py`: release pose over the rim. Straight down to x≈650. Beyond that, tilt
  forward `o=(0.7,0,-0.7)`. Reach limit ≈ 95 cm.
- `throw.py`: **untested draft** throw (planned wind-up, then a direct J2+J4 swing with a
  timed gripper open from a placeholder `RELEASE_TABLE`). Requires `--i-checked-with-staff`.
- `logger.py`: MongoDB Atlas `balltoss.attempts` (no `MONGODB_URI` → readable
  `attempt.json` in the detection-log folder, never dumped to the terminal); `suggest_offset()` = autotune from
  recent `miss_mm`. `telemetry.py`: per-move timings + snapshots in the attempt doc.
  `dashboard.py`, `stats.py`: hit-rate views.
- `detection_log.py`: per attempt, `observations/detections/<time>/` gets every frame the
  cup/ball detectors saw (raw + `_det.jpg` with candidates drawn) and `log.jsonl` (stage,
  view pose, every candidate kept/rejected with reason and world mm, survey/target/result
  events). Detectors call `detection_log.record(...)`; no-op unless `main.attempt` started a
  log. `python detection_log.py` summarizes the newest folder.
- `scripts/`, `hover_cup.py`, `scene_detection.py`: read-only/diagnostic tools.
- Docs: `TASKS.md` (owners, calibration history, log schema incl. the Phase 2 `throw`
  field), `DEMO.md`, `DASHBOARD.md`, `docs/cup-detection.md`.

## Throw work: what already exists (check before writing new code)
- `throw.py` on `main`: simple timed-release draft, not physics-based.
- `origin/rohan/tossball` → `toss_ball.py` (+ `tests/test_toss.py`): physics-based toss.
  URDF forward kinematics, ballistic release speed for a target distance, elbow+wrist
  split by ratio `K`, a runway so the joints reach full speed at the release pose, gripper
  latency compensation, and swept-path checks against table, ceiling and walls. It throws
  along +x only (no cup aiming yet).
- `sk/throw` branch (worktree at `~/.codex/worktrees/eaaf/...`): cup tracking
  (`cup_tracking.py`: remember the cup before picking) and `motion_control.py`.
- Open problems: base yaw (J0) to aim at a cup off-axis; measuring gripper open latency;
  release timing jitter over the network (15–400 ms seen); closing the loop with Atlas
  (log `throw: {...}` per TASKS.md, fit release parameters from logged misses).

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
