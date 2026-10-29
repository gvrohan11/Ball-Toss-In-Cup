# Cup detection

## What is implemented

`cup_detection/core.py` uses **YOLO11m-seg**, a larger neural instance-segmentation
model, at 1280 pixels with quarter-turn inference over the whole image. It
recognizes cups without requiring a red exterior. The old standalone color/ring
candidate generator is disabled by default. Cyan contours are learned masks;
green boxes/confidence come from the model. Classical rim fitting is used only
for optional geometry; yellow rims appear only after a validated 3D fit.
Red-body reflection fits and oversized rim ellipses are rejected.
This is a general cup model, not a Solo-specific trained model or a guarantee at
every viewpoint/occlusion. The exterior's identity cannot be verified when hidden.
The model can later be replaced by cup-specific trained weights with class
`red_solo_cup`. Confidence is the pretrained model score, not a calibrated Solo-cup probability.

`cup_detection/service.py` exposes Viam detections, paired capture, properties,
cropped point clouds, and `do_command({"command":"rim_centers"})`. Rim estimates
use aligned depth on the visible rim, robust plane/circle fitting and arc-coverage
checks. Diameter is inferred from that fit. Height is inferred only if a nearby
parallel supporting surface is visible; otherwise it is null. Missing rim depth
never falls back to the cup bottom or box center. All poses are millimeters;
serialized PCD coordinates are meters.

## Install and test locally

From the repository root, use the project's Conda environment. Its NumPy/SciPy
have been repaired using pip wheels to avoid duplicate OpenMP runtime loading.
The environment.yml now installs numeric libraries via pip consistently.
Do not use KMP_DUPLICATE_LIB_OK to hide runtime conflicts.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$PWD/.conda"
python -c "import numpy, scipy, cv2, torch; print('Conda runtime OK')"
```

First inference downloads the official pretrained weights to `.cup-models`.
Images stay local. Test the supplied images (or substitute your own paths):

```bash
python -m scripts.cup_test offline --input /private/tmp/IMG_1924.JPG /private/tmp/IMG_1925.JPG /private/tmp/IMG_1926.JPG --output /tmp/cup-photos --config config/cup-detector.thresholds.json
open /tmp/cup-photos
```

Preview the real Viam camera locally with boxes (requires the existing `.env`):

```bash
python -m scripts.cup_test live --config config/cup-detector.thresholds.json
```

Press Q/Escape to close. Test side, oblique, overhead and rotated views, cups at
different image positions, multiple cups, jackets/buckets, and an empty scene.
Capture and replay paired RGB-D for repeatable tuning:

```bash
python -m scripts.cup_test capture --frames 30 --output /tmp/cup-data
python -m scripts.cup_test offline --input /tmp/cup-data --output /tmp/cup-results --config config/cup-detector.thresholds.json
python -m pip install pytest
python -m pytest -q tests/test_cup_core.py tests/test_cup_service.py tests/test_cup_harness.py
```

Detector thresholds are `DetectorConfig` fields; put overrides in the thresholds
JSON. `rotation_search=false` speeds up inference at the cost of rotation coverage.
The photo files have no depth, so cannot test 3D output.
Rim localization remains view/depth-dependent and may return null even with a
valid cup box. This detection module does not command arm motion.

## Deploy as a Viam module and view boxes in Viam

Local preview works without deploying the custom service. To run the
service under Viam, copy this code and its environment onto the **computer
running viam-server** (a laptop path is not accessible on the robot host).

Merge `config/cup-detection.local.example.json` into the machine config; preserve
existing components/services/modules. Replace all absolute path placeholders
and set CUP_PYTHON to that host's environment. Download weights there before
starting. It adds `cup-detector-v2` and a `cup-boxes` transform camera. View
**CONTROL -> cup-boxes**. The original cup-detector and cup-3d are not replaced;
existing `cups.py`/`main.py` still use the old pipeline until explicitly integrated.

All DetectorConfig fields can be supplied as service attributes. The generated
CLI scaffold command requires `viam login`; it was attempted but unavailable
without authentication, so the module uses verified SDK EasyResource scaffolding.
No registry registration or live machine configuration was changed.

References: [YOLO11](https://docs.ultralytics.com/models/yolo11/),
[Viam vision Python API](https://python.viam.dev/autoapi/viam/services/vision/index.html).
