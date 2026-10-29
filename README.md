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
