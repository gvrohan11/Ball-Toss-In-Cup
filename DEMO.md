# 2-minute demo script — Ball Toss In Cup (sfhack11)

**Setup before going on:** ball near the robot, cup upright 70–95 cm out, Atlas dashboard on a
second screen, `python main.py --go --autotune` ready, someone on the E-stop. Backup: the video.

## 0:00 – Problem (15 s)
"Beer pong is easy for people and hard for robots: find a small ball, grab it without crushing it,
find a cup somewhere on a messy table, and land the ball in a 9 cm opening. We built that
end to end on a UFactory 850 with Viam, in one afternoon."

## 0:15 – Live run (45 s) — press enter, narrate as it moves
1. **Home → find the ball.** "The wrist camera spots the orange ball, then moves directly over it to
   measure it precisely. We calibrated the camera-to-gripper error on the real arm."
2. **Pick.** "It grabs at the ball's equator — the fingers swing down as they close, so it grabs
   high and lets them come down onto the ball."
3. **Zoom out, look across the table.** "Now it tilts back and scans for the cup at 30° and 45°."
4. **Find the cup.** "The ball in the gripper hides the cup's red side, so a red-color detector fails.
   Ours finds the cup from its opening — the inside of a red cup has a pink tint nothing else has."
5. **Drop.** "The cup is beyond straight-down reach, so it tilts the gripper forward to reach over it."
6. **Home.** Type `y`.

## 1:00 – The data (30 s) — switch to the Atlas dashboard
"Every attempt goes to MongoDB Atlas: ball and cup positions, release pose, hit or miss, and which
way it missed." Point at: **hit rate**, **miss scatter**, **failures by type**.
"And it closes the loop: with `--autotune`, the robot reads its recent misses back from Atlas and
shifts its aim to correct them."

## 1:30 – How we built it (20 s)
"Viam made the hardware software: arm, gripper, camera and motion planning behind one API, with
obstacles so it can't hit the table. Detectors and 3D vision are config blocks; our logic is ~500
lines of Python. We drove it with an AI agent through Viam's MCP server while we calibrated."

## 1:50 – The throw (live if it's landing, else video) (40 s)
`python pick_and_throw.py --aim-cup --cradle --go --i-checked-with-staff` (cup ~95 cm) or `--far`
default (80-100 cm). Pick whichever landed more consistently in the morning.
"Past ~95 cm it can't reach, so it throws. The swing was designed on the arm's kinematic model,
checked against the table and ceiling before every throw, and the speed is picked from its own
past throws in Atlas: `python throw_fit.py` shows model vs real landing per throw style."
Cups: a triangle with ~2 cm of water (stops bounce-outs), front cup ~90 cm out.

## 2:30 – What's next (10 s)
"Run the throw as a Viam module on the robot computer, so any xArm team can call `throw` with a
distance."

## Numbers to fill in before the demo
- Drop hit rate: __ / __ (from `python stats.py`; 11/11 on Oct 29)
- Throw: __ / __ ; FAR model vs real: x__ (`python throw_fit.py`)
- Furthest cup hit: __ cm
- Time per attempt: __ s (logged as `duration_s`)
