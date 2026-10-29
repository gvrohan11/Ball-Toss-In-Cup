# Ball-Toss-In-Cup
Program to make a ufactory A850 robot throw a ball into a cup

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

### Cup memory and monitored execution

`python main.py` is read-only: it reports rim candidates in the current view.
`python main.py --go` runs the revised sequence:

1. Move through two high observation poses and triangulate the nearest cup rim.
2. Save its center/radius in **world coordinates** to `observations/cup-memory.json`.
3. Locate and pick the orange ball, then measure where it sits in the gripper.
4. Transfer directly to the remembered cup with one Viam-planned move.
5. Reobserve the rim twice and remeasure the held ball before opening.
6. Return home and record the human-confirmed outcome.

During motion, JSON feedback reports stage, gripper pose, payload position,
horizontal distance to the remembered cup and holding state. Polling targets
roughly 2 Hz subject to network latency. It is written to
`observations/run-feedback.jsonl`. Lost payload or failed telemetry stops the arm.
World coordinates are remembered even when the cup leaves the camera view;
**release still requires a fresh stationary visual check**. The code refuses to
release if memory is older than 180 seconds, the cup moved over 10 mm, association
is ambiguous, or payload alignment exceeds 8 mm. A blocked release holds position
with the gripper closed; it does not chase a moved cup.

Rim tracking currently targets upright, approximately circular white rims on
nearby cups (world x<=650 mm), using the existing wrist-camera calibration.
Both observation poses must see the cup. Fully clipped rims and unreliable
stereo matches are rejected. These limits intentionally exclude the earlier
far-cup tilted-release heuristic from this revised loop.

The local runner removes conversational delays and the post-pick search scan.
It keeps pauses for observation, grip and release. Transfer is one planned move,
with driver speed 10 deg/s and acceleration 20 deg/s². This is **not** a blended
multi-waypoint trajectory: Viam's Python `Move` accepts a single destination;
multi-goal planning needs an in-process Go planner. Joint streaming is not used
because it bypasses obstacle planning. See
[Viam multi-waypoint planning](https://docs.viam.com/motion-planning/move-an-arm/multiple-waypoints/).

Hardware retry on October 29 verified cup memory before picking, a successful
pick/lift, and a single planned loaded transfer with live feedback. The original
release check rejected a clipped rim. Raising the ball release height from
140 to 165 mm restored the full view; recovery passed both visual checks with
5.2 mm payload alignment error and put the ball in the cup. The arm returned
home open. This was not an uninterrupted full-cycle pass; repeat the entire
sequence with the updated release height before claiming one. Results and image:
`observations/automated-release-result.json`, `observations/automated-release-success.jpg`.

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

`observations/frame-calibration.json` records the configured camera/arm/gripper
relations and the successful live gripper-to-camera transform. Both camera and
gripper are children of the arm flange. Camera translation is (83, -14, 18) mm;
gripper translation is (0, 0, 105) mm. The imported camera rotation is nontrivial
and must be included, rather than just subtracting translations.
The user reports verified calibration; no measurement/residual report was found
in the repository. `observations/orange-grasp.offline.json` is an analysis of an
older sample and must not be used for live motion.

YOLO configuration is included in `config/yolo.proposed.json` as a proposed setup:
`ball-yolo` uses `yolov8n.pt`, restricted to the `sports ball` class at 0.25
confidence, and `ball-yolo-3d` projects those detections using RealSense depth.
The module runs on the Viam machine, so local PyTorch installation is unnecessary.
Test the team's world-coordinate detection using YOLO (read-only):

```bash
BALL_VISION_SERVICE=ball-yolo-3d python balls.py
```

Installation was attempted on October 29, but the module failed to unpack
`PIL/_avif.cpython-311-x86_64-linux-gnu.so` (PyInstaller decompression error).
The failed additions were removed from the live machine. The proposal JSON
includes the YOLO services; merge its entries only after resolving that module
installation error. The command above requires those services to be running.
The default remains `ball-3d` until YOLO has been validated on the ping-pong ball.

`ball-detector` is tuned for the orange ping-pong ball under the current lighting.
`ball-3d` uses the existing detections-to-segments module and RealSense depth.
Color matching alone does not validate that an object is a ball or provide a
safe grasp; verify the 3D dimensions and reference frame before motion.

Copy `.env.example` to `.env` and fill in sfhack11's CONNECT-tab credentials.
With the Conda environment active, run:

```bash
python scripts/inspect_ball.py
```

This prints 2D detections, compact 3D geometry, and unvalidated example contact
points in world coordinates. It saves the latest result in
`observations/ball.latest.json` without moving the arm or gripper.
SDK access avoids the MCP tool's 64 KB point-cloud response limit.
The October 29 measurement put the center near (310, 7, 26) mm in world
coordinates, with dimensions about 41 x 36 x 39 mm. The configured table top
is at z=-23 mm; a ball resting there should have a center around z=-3 mm.
Resolve this discrepancy and verify the gripper TCP before attempting a pick.
`pick_ball.py` now contains the team's grasp sequence and measured corrections;
see `REVIEW.md` for its assumptions and the fixes made on the review branch.
The newer sequence has not been run on hardware during this review.

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
