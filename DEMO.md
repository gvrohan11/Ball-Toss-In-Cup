# 5-minute demo — Ball Toss In Cup (sfhack11)

Judging: functionality 40% · ambition 20% · creativity 20% · modularity 15% · Halloween 5%.

## Before going on (10 min before)
- **Robot (SSH window):** `cd ~/Ball-Toss-In-Cup && . .venv/bin/activate && git pull`
- **Laptop (Command Prompt, NOT SSH):** `python dashboard_web.py --hours 1` → second screen.
  (`--hours 1` = the board starts clean-ish; it fills live during the demo.)
- 6 red cups on the taped ARC (~92 cm from the base), ~3 cm water, **dry** orange balls 30-55 cm
  in front of the base, ~8 cm apart, nowhere near the cups.
- Someone on the E-stop. Backup: the 6-for-8 video.
- Known-good code: branch `six-cups-oct30` (`git checkout six-cups-oct30` if anything breaks).

## 0:00 – The problem (30 s)
"Beer pong is easy for people and hard for robots. Find a 4 cm ball on a table, pick it up, find six
cups, and throw it into a 9 cm opening a metre away. Everything's a separate hard problem: vision,
grasping, aiming, and throwing with an arm that was built to move slowly and precisely."

## 0:30 – Live: fill the cups (2 min) — start it, then narrate
```bash
python pick_and_throw.py --fill-cups --go --i-checked-with-staff
```
1. **Survey.** "It looks at the table from six angles and merges what it sees into six cups — numbered
   right to left — and remembers them. It never re-targets a filled cup."
2. **Find + pick.** "Wrist camera finds the orange balls, moves over one and takes three close-up
   readings — they agree to a tenth of a millimetre — then picks it."
3. **Throw.** "The base turns to aim, the arm whips up and the fingers open mid-swing."
4. Answer honestly (`in`, `rim front`, `short 3`…). Point at the dashboard: the cup turns green.
Let 3-4 balls go live, then move on while it keeps going.

## 2:30 – How it throws (ambition + creativity, 1 min)
- "The swing was designed on the arm's kinematic model and is checked against the table and ceiling
  before every throw — fast throws skip Viam's planner, so we do the safety check ourselves."
- "Timing is everything: the fingers take ~0.15 s to react, the swing lasts 0.7 s. From the laptop,
  Wi-Fi added up to 400 ms of random delay — 0 for 73 on Thursday. We moved the code onto the robot's
  own computer: 3 ms. Same code, from zero hits to filling all six cups."
- "We tried four throw styles. Open-hand throws physically can't work on this arm — it can't brake
  hard enough to fling a loose ball. We proved that on the model, so we stopped trying them."

## 3:30 – MongoDB: the robot learns (1 min) — dashboard, right chart
"Every attempt goes into Atlas: where it aimed, how fast it swung, where the ball landed. The right
chart is the arm's physics model (dashed) against reality. Balls fly ~12% shorter than the model —
air drag on a 2.7 g ball. Before every throw the robot reads its last 8 throws from Atlas and picks
its speed from the learned line. This morning it was missing 4-5 cm short; two runs later it was
throwing 6 for 8."
Point at: **Throws in the cup**, **cups filled**, **×0.87 learned**, the feed.

## 4:30 – Modularity + Viam (30 s)
"Viam turned the hardware into software: arm, gripper, camera and motion planning behind one API.
Cup detection also runs as a Viam vision service. The throw is a preset table + a learned
correction — any xArm team can give it a cup distance."

## Numbers (fill in right before)
- Throws: **6 cups filled in 8 throws (75%)** on the last full run, every throw at the target cup.
- Drop (pick → carry → drop): 11/11 when the cup is in reach.
- Learned correction: ×____ (dashboard).  Today's throws in: ____ / ____ (dashboard).
- Command latency: laptop 15-400 ms → robot computer 2.6 ms (jitter 0.7 ms).

## If something goes wrong live
- Ball not picked → it retries, then moves on. Keep talking; it's the honest demo.
- E-stop / "Emergency Stop Button Pushed In" → see WORKING.md (clear_error), restart the command.
- Robot dead → play the video, walk through the dashboard.
