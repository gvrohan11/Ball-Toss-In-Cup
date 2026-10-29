# Ball and cup integration review

Reviewed origin/main at `94d76de` on October 29, 2026. Working branch:
`codex/review-ball-cup`. The remote ball/detection branch has no additional
commits beyond main; the pick/drop implementation was merged from manan/integration.

Retained the new two-view ball localization, camera-centering step, gripper
settling, angled cup scanning, and planned release moves. Local detection tools,
observations, frame reports and environment-variable compatibility were preserved.

Fixed on this branch:

- Stop the sequence when motion returns false; do not release after a failed move.
- Reject missing/invalid gripper positions and fail when closing never settles.
- Return a failed pick before lifting if the gripper closed on nothing.
- Reject camera/table intersections behind the camera.
- Reject invalid focal lengths and nonfinite ball coordinates.
- Keep dry-run attempts out of MongoDB.

Calibration evidence newly available in the integration code:

- `pick_ball.py` uses a measured world-XY correction of (-15, +20) mm at wrist
  theta=180, plus a fixed gripper grasp height of 70 mm. The header reports
  successful manual trials; these were not reproduced during this review.
- `cup_aim.py` uses tabletop z=-10 mm, 120 mm cup height, 30 mm base radius,
  and a measured correction (-56, +17) mm. The configured obstacle table top
  in the earlier snapshot is -23 mm. Reconcile these references before trusting
  clearance calculations or changing the hardware setup.
- The gripper frame origin is not automatically the finger contact center.
  The fixed 70 mm pick height accounts for finger behavior on this setup;
  an estimated sphere center must not be passed directly as the gripper pose.

Remaining limits:

- Ball range is inferred from apparent diameter; partial orange masks or glare
  change this estimate. Only the largest roughly square detection is selected.
- Camera and gripper XY offsets are hardcoded for wrist rotations 0 or 180.
- Cup location is a red-body/table-plane heuristic rather than a measured rim.
  Fixed corrections may not transfer to other cups, positions or camera mounts.
- Holding is inferred from finger position, not independently verified ball contact.
- Dry-run scans do not move the camera, so all views use its current pose.

Validation: offline regression tests in `tests/test_sequence_guards.py`, syntax
checks and diff whitespace checks. No robot motion or machine config edits were
performed for this review.
