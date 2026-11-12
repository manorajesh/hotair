# Hot Air pipeline

Multi-phone retroreflective shadowgraph → synced clips → a 4D neural field of the air.

You film the same moment with several phones (flash on, pointed through the heat at a retroreflective sheet). The pipeline:

1. **sync**: lines the phones up from their audio (a clap is best, but room noise works).
2. **calibrate**: works out each phone's focal length and position from a ChArUco board clip.
3. **process**: turns each phone's footage into "air" and "heat" videos plus side-by-side clips, and tracks the flame.
4. **place**: phones that never read the board are positioned from the flame seen by all the others.
5. **field**: fits a 4D neural field (x, y, z, t → heat) that agrees with every phone at once.
6. **render**: orbiting 3D video, height slices, a seen-vs-predicted check, and a training-progress clip.

## Install (Mac, M-series)

```bash
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On a Mac the 3D step runs on the GPU automatically (PyTorch "mps"); video decoding uses the hardware decoder.

## Run

1. Put your clips in a folder with a `project.json` (copy `project.example.json`).
2. Edit the file names, the board's square size, and the time windows (see below).
3. Run:

```bash
python hotair.py all path/to/project.json
```

Or one step at a time: `sync`, `calibrate`, `process`, `place`, `field`, `render`. Each step saves its results, so you can re-run later steps after changing settings. For example, after changing `field.steps`, run `field` and then `render`.

## project.json

```jsonc
{
  "reference_camera": "C",                    // all times below are on THIS camera's clock
  "cameras": {                                // any number of phones; short names
    "L": {"take": "stage_left_lighter.MOV",  "calib": "stage_left_calib.MOV"},
    "C": {"take": "center_lighter.MOV",      "calib": "center_calib.MOV"},
    "R": {"take": "stage_right_lighter.MOV", "calib": "stage_right_calib.MOV"}
  },
  "board": {"squares_x": 8, "squares_y": 5, "square_mm": 28.36,   // measured black square on your screen
            "marker_ratio": 0.75, "dictionary": "DICT_5X5_100"},
  "calibration": {"detect_fps": 15, "same_lens": true},   // same_lens: all phones are the same model
  "take": {
    "reference":    [4.5, 6.2],               // seconds with the sheet empty and nothing hot in view
    "analysis":     [12, 80],                 // the part to process into clips
    "highlight":    [17, 38],                 // the part to also export as short side-by-side clips
    "field_window": [40, 46]                  // the part to reconstruct in 3D (keep it to ~5-10 s)
  },
  "processing": {"clip_width": 960, "clip_fps": 30, "field_width": 480, "field_fps": 15,
                 "air_contrast": 0.04, "heat_contrast": 0.02},
  "field": {"box_size_m": [0.26, 0.45, 0.26],  // width, height, depth of the 3D region (flame at the bottom)
            "steps": 1500, "device": "auto", "carve_threshold": 0.002},
  "output": "out"
}
```

Finding the times: watch the centre (reference) phone's clip. `reference` is a stretch before anyone walks in or anything hot appears. `field_window` is a steady stretch where the heat is visible in every view.

## What you get (in `out/`)

| File | What it is |
|---|---|
| `sync_check.mp4` | First 10 s of all phones side by side. Movements should line up exactly. |
| `calibration_detections.png` | Board corners found by each phone. |
| `calibration.json` | Focal length and position of every phone (metres, in the anchor phone's coordinates). |
| `cameras/<cam>/side.mp4` | Original and processed air side by side, whole analysis window. |
| `cameras/<cam>/side_highlight.mp4` | The same, highlight window only. |
| `cameras/<cam>/air.mp4` | Processed shadowgraph only. |
| `cameras/<cam>/heat.mp4` | Heat magnitude in colour (what the 3D step uses). |
| `cameras/<cam>/original_synced.mp4` | Untouched footage cut to the shared clock. |
| `cameras/<cam>/strongest.png`, `reference.png` | Strongest frame, and the empty reference. |
| `all_cameras_*.mp4` | Grids of all phones in sync: air, heat, side_highlight, original_synced. |
| `field/orbit.mp4` | The 3D plume playing while the view circles it (cleaned). |
| `field/orbit_raw.mp4` | The same before cleaning (shows the limits of the camera layout). |
| `field/slices.mp4` | Horizontal cuts through the plume at several heights, seen from above. |
| `field/check_seen_vs_predicted.mp4` | Each phone's heat map next to what the 3D field predicts it should see. |
| `field/training_progress.mp4` | The field's predictions sharpening during training. |
| `field/volume.npy` | The cleaned 4D field as an array: (time, x, y, z), 5–6 mm voxels (`volume_raw.npy`: before cleaning). |
| `log.txt` | Everything printed, including fit quality numbers. |

## Shooting tips that matter most

- **Spread the phones wide.** ±45° on an arc beats ±15°: depth is the weakest direction with a narrow spread. Bend the sheet so every phone sees it near head-on.
- **Calibrate close.** Bring the board within about 1.5 m of each phone, and hold each pose still for 2 s. Shoot calibration in 4K if you can.
- **Clap once** at the start of every take and calibration clip.
- **Don't move anything** between calibration and the takes.
- **Record the empty sheet first** for a few seconds: that's your `reference`.
- **Lock focus and exposure** (long-press "AE/AF lock"), flash on, landscape.

## Single phone

`tools/single_camera_bos.py` processes one clip (BOS or shadowgraph) without calibration:
`python tools/single_camera_bos.py clip.MOV --ref 4 6`.

`tools/make_charuco.py` makes the calibration image for a laptop screen.

## How the 3D part works (short version)

Every heat-map pixel is treated as the sum of "heat" along that camera's line of sight through a box above the flame. A 4D K-Planes neural field (six 2D feature planes over xy, xz, yz, xt, yt, zt, multiplied together, then a small network) is trained so that simulated sums match all phones at once, with smoothness in space and time. When rendering, the result is cleaned (soft space carving): density fades out wherever any phone saw no heat along its line of sight, and nothing is kept below the flame. `field.carve_threshold` (default 0.002) sets how strict this is; changing it only needs `render`, not retraining.

The result is a **shape field** (where the heat is and how it moves), not calibrated temperature.

## Run times (for reference)

On the 2-core cloud machine used for testing, with three 1080p phones: sync about 1.5 min, calibration about 7 min (cached afterwards), processing 14 s of footage about 2 min, 300 training steps about 7 min. An M4 Pro should be several times faster, and the 3D step uses its GPU.

## Troubleshooting

- **"weak match" during sync**: that phone's audio didn't line up clearly. Clap in view next time. You can check with `sync_check.mp4`.
- **A phone reads the board in very few frames**: it was too far from the board, or the screen was too dim. That phone gets placed from the flame instead (`place` step), which works if the other phones are board-calibrated.
- **Halos or streaks in heat.mp4**: the sheet is moving or wrinkled. Tape it flat.
- **3D looks stretched front to back**: the phones are too close together in angle. Spread them wider.
- **Nothing in the 3D box**: the box is placed above the flame automatically. If you're not using a flame, set `field.box_center_m` (metres, in the anchor phone's coordinates; see calibration.json).
