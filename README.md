# Ball-Toss-In-Cup
Program to make a ufactory A850 robot throw a ball into a cup

## Quick start (what we demo)

```
pip install viam-sdk==0.85.1 python-dotenv pymongo opencv-python numpy scipy
cp .env.example .env        # fill VIAM_ADDRESS / VIAM_API_KEY / VIAM_API_KEY_ID / MONGODB_URI
python check_connection.py  # read-only check
python home.py              # send the arm to the saved safe home pose
python main.py              # dry run: prints the plan, moves nothing, logs nothing
python main.py --go         # live: home -> survey + remember all cups -> find + pick ball (slowly) -> drop -> home
python pick_and_throw.py --aim-cup            # Phase 2 dry run: find cup, plan + safety-check the throw
python pick_and_throw.py --aim-cup --go --i-checked-with-staff          # FAR throw at the cup (80-100 cm)
python pick_and_throw.py --aim-cup --power --go --i-checked-with-staff  # POWER throw for 1-2 m (direct mode)
python pick_and_throw.py --aim-cup --cradle-check --go --i-checked-with-staff  # spoon pose, NO swing: does the ball stay?
python pick_and_throw.py --survey-only --cradle --go --i-checked-with-staff    # find + list the cups only
python pick_and_throw.py --aim-cup --cradle --go --i-checked-with-staff        # spoon throw, nearest cup in ~93-102 cm
python pick_and_throw.py --fill-cups --cradle --go --i-checked-with-staff      # one ball per cup; filled cups skipped
python throw_fit.py              # how far each throw preset really goes vs its model (from Atlas)
python scripts/latency_check.py  # command round-trip; run on laptop AND robot (docs/RUN_ON_ROBOT.md)
python throw_safety.py           # swept-path check of every throw preset, no robot
python main.py --go --autotune   # also correct the aim from recent misses stored in Atlas
python main.py --go --runs 10    # 10 attempts in a row (cups surveyed + remembered first), prints N/10
python main.py --go --cup 2      # several cups: drop into cup 2 of the survey list (1 = nearest)
python main.py --go --fill-cups  # several balls + stationary cups: one ball per cup, filled cups discarded
python main.py --go --ball white # pick a WHITE ball instead (or --ball any); default is orange. Put it alone on flat table, not in a bowl
python detection_log.py          # what the camera saw + detected in the last attempt (observations/detections/)
python dashboard_web.py          # LIVE demo dashboard from Atlas on http://localhost:8765 (run on the laptop, second screen)
python dashboard.py --watch      # older static hit-rate page
python -m pytest -q tests        # offline tests, no robot
```

How it works: [TASKS.md](TASKS.md) (status, calibrations), [DASHBOARD.md](DASHBOARD.md)
(Atlas), [DEMO.md](DEMO.md) (2-minute demo script).

Safety: every run starts at the home pose, checks the gripper first, uses Viam motion
planning (table/wall/ceiling obstacles), and if anything fails while holding the ball it
puts the ball back and goes home. Someone stays on the E-stop.

## Python environment

Uses Python 3.11 and the Viam Python SDK to access the arm through Viam,
plus NumPy/SciPy for trajectory calculations, Matplotlib for plots, and OpenCV
for camera processing. The UFACTORY SDK is also installed for direct SDK work.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda env create --prefix "$PWD/.conda" --file environment.yml
conda activate "$PWD/.conda"
```

Check the installation without connecting to the robot:

```bash
python -c "from viam.robot.client import RobotClient; from viam.components.arm import Arm; import numpy, scipy, matplotlib, cv2, dotenv; print('Environment ready')"
```

To update the environment after changing dependencies:

```bash
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda env update --prefix "$PWD/.conda" --file environment.yml --prune
```

Environment setup does not connect to the arm or enable motion.

## Viam

Team machine: **sfhack11**, main part `sfhack11-main`.
Machine ID: `695800a5-6191-4136-ab54-968225e34645`.

Shared configuration snapshots (exported October 29, 2026):

- `config/sfhack11.json`: exact saved main-part configuration, with fragment imports.
- `config/sfhack11.standalone.json`: the same configuration with the two resolved
  fragments inlined, for teammates who cannot access the original fragments.

Both include ball/cup vision services, the arm, camera, gripper, and workcell
obstacles (the hardware and some obstacles are imported in the first file).
The standalone file is intended for Viam's machine CONFIGURE JSON editor;
adapt the arm IP, camera identity, and frame calibration for a different setup.
Registry modules still require network access and track `latest`, as in the
source machine. Snapshots contain no API keys. Exporting these files does not
apply changes to the live machine.

The Viam MCP server is already connected in the development assistant. It is a
remote OAuth service at https://app.viam.com/mcp; it does not run inside Conda.
Teammates connect their own Viam accounts in their MCP clients.
The `viam-sdk` package in Conda is for project Python code that talks to Viam.

Get the machine address and credentials from its CONNECT tab in the Viam app.
Keep API keys in a local `.env` file (ignored by Git), never in shared machine
configuration JSON. Each teammate should use credentials they are authorized to use.

## Connecting to the robot

1. `cp .env.example .env` and fill in `VIAM_ADDRESS`, `VIAM_API_KEY`, `VIAM_API_KEY_ID`
   from sfhack11 -> CONNECT -> Python (turn on "Include API key").
2. `python check_connection.py` - read-only: lists resources, arm pose, ball/cup detections.
3. In your own code: `from connection import connect` then `robot = await connect()`.

`pick_ball.py` is a dry run by default; `python pick_ball.py --go` moves the arm
(someone on the E-stop).

References: [Viam Python setup](https://docs.viam.com/build-apps/setup/python/)
and [Viam MCP server](https://docs.viam.com/reference/mcp/).

## Read-only vision tools
`white_ball.py` / `scene_detection.py` find white balls from depth (used by `--ball white`).
Saurabh's plate / grasp inspection scripts (`scripts/inspect_plate.py`, `orange_grasp.py`,
`inspect_ball.py`) were removed from main on Oct 29 (nothing used them); they're on the
`saurabh/read-only-tools` branch and in git history.
