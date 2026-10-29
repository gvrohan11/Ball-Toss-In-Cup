# Ball-Toss-In-Cup
Program to make a ufactory A850 robot throw a ball into a cup

## Quick start (what we demo)

```
pip install viam-sdk==0.85.1 python-dotenv pymongo opencv-python numpy scipy
cp .env.example .env        # fill VIAM_ADDRESS / VIAM_API_KEY / VIAM_API_KEY_ID / MONGODB_URI
python check_connection.py  # read-only check
python main.py              # dry run: prints the plan, moves nothing, logs nothing
python main.py --go         # live: home -> find + pick ball -> scan for cup -> drop -> home
python main.py --go --autotune   # also correct the aim from recent misses stored in Atlas
python dashboard.py --watch      # live hit-rate dashboard from Atlas (second screen)
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

## Saurabh's read-only vision tools (never move the arm)

These only read the camera and arm pose; the main loop does not use them yet.
They need `scipy` (and `requirements-yolo.txt` only for the optional YOLO backend).

### Inspect the orange ball

Read-only grasp analysis is in `scripts/orange_grasp.py`. It reads aligned color
and point-cloud data, checks that arm joints remain stationary, detects round
orange regions, fits a known-radius 20 mm sphere, and estimates a nearby support
plane. It saves annotated images and candidate contact points. It never moves
the arm. Candidate contacts are object geometry, not calibrated gripper TCP
commands; the finger closing axis and contact-center offset still need verification.

```bash
python scripts/orange_grasp.py
```

### Multicolor balls on a plate (read-only CPU prototype)

For known 40 mm ping-pong balls, the geometry backend uses color-independent
circle proposals followed by a fixed-radius sphere fit in aligned RGB-D data.
It needs no neural model or GPU. Run with the plate's image rectangle:

```bash
.conda/bin/python scripts/inspect_plate.py --backend geometry --plate-roi 570 215 920 565
```

The example rectangle is specific to the captured setup; update it when the
plate or camera moves. The script reads the camera and arm pose, transforms
accepted centers through Viam's configured frame tree, rejects arm movement
during capture, and saves images and JSON under ignored `observations/`.
It never moves the arm. `grasp_ready` and `motion_ready` remain false.
A fitted sphere is a geometric candidate, not a semantic guarantee: plate
patterns, occlusion, invalid depth, and other spherical objects need validation.
The current saved scene yields four candidates matching the two white and two
orange balls. This is a single-scene check, not validation of arbitrary piles.

Do not feed these coordinates directly into the current orange-only picking
loop. Plate/rim clearance, gripper opening and approach direction, neighbor
collisions, and held-ball verification for other colors still need integration.
The reported neighbor gaps are diagnostics, not a collision-free grasp plan.

The optional YOLO backend is for comparison and cup/ball experiments:

```bash
.conda/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.conda/bin/python -m pip install -r requirements-yolo.txt
YOLO_CONFIG_DIR="$PWD/.yolo-config" MPLCONFIGDIR=/tmp/ball-toss-mpl .conda/bin/python scripts/inspect_plate.py --backend yolo --plate-roi 570 215 920 565
```

YOLO inference explicitly uses CPU. Its default pretrained YOLO11 segmentation
model missed the ping-pong balls in the full-frame check; a diagnostic crop
classified orange balls as fruit and missed white balls. It is therefore not
the default plate-ball detector. If a learned detector becomes necessary,
fine-tune with labeled images of this setup on a GPU, then validate CPU inference
on the deployment machine. No fine-tuning has been performed.
