# MongoDB Atlas: setup, logging, dashboard

Data: database `balltoss`, collection `attempts` (written by `logger.py`).

## 1. Get the connection string (10 min)
1. Every team got an Atlas cluster for the hackathon. Ask a MongoDB/Viam staff member which
   cluster is ours and to invite your email, or create a free M0 cluster at cloud.mongodb.com.
2. Atlas → **Database Access** → add a database user (username + password).
3. Atlas → **Network Access** → add your current IP (or `0.0.0.0/0` for the weekend only).
4. Atlas → **Connect** → **Drivers** → Python → copy the `mongodb+srv://...` string,
   put the password in, and add it to `.env`:
   ```
   MONGODB_URI=mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/?retryWrites=true&w=majority
   ```
5. `pip install pymongo` (or update the conda env), then `python logger.py`.
   It should print `[log] saved attempt ...`.
6. `python stats.py --seed-today` stores today's 5 successful drops (once only), then
   `python stats.py` prints the hit rate.

## 2. Logging during runs
`python main.py --go` logs every attempt automatically. After each drop it asks:
- `Did it go in? [y/n]`
- on a miss: `cm past(+)/short(-), cm left(+)/right(-)` **as seen from the robot**, e.g. `2,-1`.
  That miss direction is what powers the auto-tune.

`python main.py --go --autotune` shifts the aim by the average of the recent misses
(capped at 3 cm). That's the "robot learns from its own data" story.

## 3. Dashboard (Atlas Charts)
1. Atlas → **Charts** → Add dashboard "Ball Toss".
2. Data source: our cluster → `balltoss.attempts`.
3. Charts to add:
   - **Hit rate** (Number chart): filter `{ "result.hit": { "$in": [true, false] } }`,
     aggregate `result.hit` as mean (true = 1). Show as %.
   - **Attempts over time** (Line): x = `timestamp` (binned by 5 min), y = count, series = `result.hit`.
   - **Miss scatter** (Scatter): x = `result.miss_mm.x`, y = `result.miss_mm.y`, color = `session`.
     The cup is at (0, 0); the cloud of misses shows the aim bias.
   - **Failures** (Bar): x = `result.failure`, y = count.
   - **Cup positions** (Scatter): x = `cup.pos_mm.0`, y = `cup.pos_mm.1`, color = `result.hit`.
4. Set auto-refresh to 10 s and put it on a second screen during the demo.

Judges' angle: `suggest_offset()` reads the recent misses back out of Atlas and returns
a correction for the next drop - the robot tunes its own aim from its logged data.

## 4. Telemetry on each attempt (`telemetry` field)
Every live attempt also stores a timeline, so misses can be explained and the throw tuned:
- `telemetry.events[]`: `t` (s from start), `stage` (home, ball_measured, after_pick,
  cup_view_N, released…), `joints_deg` (6 joints), `gripper_pose` (world x/y/z + orientation),
  `finger_pos`, plus stage details (ball/cup position, which scan view found the cup).
- `telemetry.moves[]`: every planned move with target, success and seconds.
- `telemetry.phases`: seconds spent in start / find_ball / pick / cup_scan / drop.
- `telemetry.images[]`: small JPEG thumbnails of the ball view and the cup view.
- `telemetry.config`: corrections, heights, autotune flag and git commit used for the run.

Example queries (Atlas Data Explorer): average scan time
`[{ $group: { _id: null, s: { $avg: "$telemetry.phases.cup_scan" } } }]`;
which scan view finds the cup: filter `telemetry.events.found: true`.
