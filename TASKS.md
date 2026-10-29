# Who does what

One arm, four people. Everyone builds **one module** with **one function**; `main.py` glues them together.

**Shared rules**
- Positions are **mm in the `world` frame**: `(x, y, z)`. The arm base is (0, 0, 0).
- The gripper points **straight down** (`o_z = -1`) unless the throw needs otherwise.
- Use **motion planning** (`motion.move` on `"gripper"`) so obstacles are respected.
- **Say it out loud** before saving a config change in the Viam app.
- **Someone on the E-stop** whenever the arm moves.
- Work on your own branch, open a PR into `main`.

## Manan: lead and integration
- Files: `pick_ball.py`, `main.py`
- Get `pick_ball.py` working (dry run, then `--go`).
- Turn it into `find_ball(robot)` and `pick(robot, ball)`.
- Own `main.py`: look → find ball → pick → find cups → drop → log → repeat.
- Own the Friday demo flow.

## Teammate 1: cup detection
- File: `cups.py`, which provides `async def get_cups(robot) -> list[tuple[float, float, float]]`
- From straight above, an upright cup only shows its white inside, so the red detector fails.
  Find an **angled look pose** where the camera sees the red sides.
- Tune `cup-detector` (2D) and `cup-3d` (3D) in the Viam app.
- Return each cup's **rim center** in the world frame.
- Backup if color fails: YOLO "cup" class (ask Viam staff or Claude).

## Teammate 2: drop, then throw
- File: `drop.py`, which provides `async def drop_in_cup(robot, cup) -> dict`
- With the ball already in the gripper: move above `cup`, open the gripper, and return to a safe pose.
- Return what you did, e.g. `{"release_pose": {...}, "height_mm": 150}`.
- Measure hit rate and tune the height and x/y offset.
- Stretch goal: a throw from a distance (fast joint-space swing). Clear it with Viam staff first.
  Direct joint moves ignore obstacles.

## Teammate 3: MongoDB Atlas
- File: `logger.py`, which provides `def log_attempt(doc: dict) -> None`
- Set up the event's Atlas cluster. Put the URI in `.env` as `MONGODB_URI` and add it to `.env.example`.
- Store every attempt: ball pos, cup pos, release pose, hit or miss, timestamp, and optionally an image.
- Dashboard: attempts, hit rate, miss positions.
- For the judges: query past misses to suggest a release offset.

## Timeline
- **Thu, before 9 PM close:** pick works, `get_cups()` returns positions, `log_attempt()` writes to Atlas.
- **Fri, by noon:** full loop works end to end, dashboard is live.
- **Fri 3:30 PM:** demo. Throw only if there's time.
