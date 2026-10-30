# What works (Oct 30, 10:10) — don't lose this

**Result:** `--fill-cups` from the robot computer threw 2 balls into cups 1 and 2 of a 6-cup arc
(2 for 2), fully automatic: survey → find ball → pick → aim → throw → next cup.
Exact code: branch **`throw-works-oct30`** (`git checkout throw-works-oct30` to get it back; it never changes).

## The command
```bash
ssh viam@192.168.31.142                     # robot computer (password from staff)
cd ~/Ball-Toss-In-Cup && . .venv/bin/activate
git pull
python pick_and_throw.py --fill-cups --go --i-checked-with-staff
```
On the tagged code the same run needs `--power --partial`. On `main` after the lock-in PR,
POWER and the partial open are the defaults. Answer every throw in plain words:
`in`, `in 3`, `rim`, `rim 4`, `-5`, `+8,3`, `drop`.

## Setup that worked
- **Cups:** 6 red solo cups on an ARC ~92 cm from the centre of the robot base, 2-3 cm apart,
  from ~23° right to ~22° left. Surveyed positions (x, y mm):
  (852,-369) (891,-239) (918,-84) (912,+60) (881,+206) (855,+350). Numbered right → left.
- **Balls:** orange, 30-55 cm in front of the base, a few cm apart, nowhere near the cups.
  One ball per cup; it stops when it can't find a ball.
- **Run from the robot computer over SSH**, not the laptop: commands take ~3 ms there
  (`scripts/latency_check.py`: median 2.6 ms, jitter 0.7 ms) vs 15-400 ms from the laptop.

## The numbers behind it
- POWER throw (`throw.POWER`): direct mode, fingers back/down, slow swing ~78-85 deg/s for 92-95 cm.
- Release: partial open (+150 units at gripper speed 5000) sent ~0.27 s into the swing.
  Finger reaction on the robot: 0.10-0.18 s after the command (`scripts/gripper_timing_test.py --partial`).
- Distance: `throw_fit` learned from Atlas that balls fly **x0.915** of the URDF model (air drag),
  from 6 measured throws at 68-85 deg/s (-10, -5, -3, -1 cm, rim, in). It picks the speed itself.
- Aim: base yaw = atan2(y, x) of the cup. Correct direction, checked on cups 23° and 15° right.

## What did NOT work (don't retry blindly)
- FAR / open-hand / spoon (cradle) throws: the arm (max 1145 deg/s²) can't brake hard enough to
  get a loose ball out of the hand; it dribbles out at the end of the swing.
- Fingers-up cradle: infeasible for the same reason (scripts/design_cradle.py).
- Laptop-driven throws: release timing was random (Oct 29: 0 hits in 73 throws).

## If something goes wrong
- **"Emergency Stop Button Pushed In"**: twist the red button up, then clear the error:
  ```bash
  python - <<'EOF'
  import asyncio
  from connection import connect
  from viam.components.arm import Arm
  async def m():
      r = await connect(); a = Arm.from_robot(r, "arm")
      print(await a.do_command({"clear_error": True})); print(await a.do_command({"get_error": True})); await r.close()
  asyncio.run(m())
  EOF
  ```
- **"no cup ... in the band"**: cups moved. Run `python pick_and_throw.py --survey-only --go --i-checked-with-staff`.
- **Lands consistently short/long**: keep answering honestly; the fit adapts after ~3 throws.
  To force it: `--speed 82` (≈1.3 cm per deg/s at this distance).
- **New SSH window**: always `cd ~/Ball-Toss-In-Cup && . .venv/bin/activate` first.
