# MongoDB Atlas dashboard (Atlas Charts)

Data: database `balltoss`, collection `attempts` (written by `logger.py`).

1. Atlas → **Charts** → Add dashboard "Ball Toss".
2. Data source: the event cluster → `balltoss.attempts`.
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
