# Hot Air pipeline

Multi-phone retroreflective shadowgraph → synced clips → a 4D neural field of the air.

You film the same moment with several phones (flash on, pointed through the heat at a retroreflective sheet). The pipeline:

1. **sync**: lines the phones up from their audio (a clap is best, but room noise works).
2. **calibrate**: works out each phone's focal length and position from a ChArUco board clip.
3. **process**: turns each phone's footage into "air" and "heat" videos plus side-by-side clips, and tracks the flame.
4. **place**: phones that never read the board are positioned from the flame seen by all the others.
5. **field**: fits a 4D neural field (x, y, z, t → density) whose rendered shadowgrams match every phone's at full resolution.
6. **render**: an orbit (density and the shadowgram a phone there would film), height slices, a seen-vs-predicted check, and a training-progress clip.

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
  "processing": {"clip_width": 960, "clip_fps": 30,
                 "field_width": null, "field_fps": 30,  // frames the 3D step fits (null: native resolution)
                 "air_contrast": 0.04, "heat_contrast": 0.02},
  "field": {"box_size_m": [0.26, 0.45, 0.26],  // width, height, depth of the 3D region (flame at the bottom)
            "steps": 2000, "device": "auto"},   // 12000 steps gives a much sharper fit (~40 min on an M4 Pro)
  "output": "out"
}
```

Finding the times: watch the centre (reference) phone's clip. `reference` is a stretch before anyone walks in or anything hot appears. `field_window` is a steady stretch where the heat is visible in every view.

Other `field` settings, all optional:

| Key | Default | What it does |
|---|---|---|
| `vox_mm` | `[10, 5, 2.5, 1.25]` | Feature-plane cell size at each scale (finest last). |
| `channels` | `8` | Features per plane. More is sharper and slower. |
| `patches_per_camera` | `12` | 32×32 px patches rendered per phone per step. More is a steadier fit and slower. |
| `samples_per_ray` | `48` | Samples along each line of sight during training. |
| `highpass_px` | `30` | Removes slow lighting changes from the shadowgrams (in 1920-px-wide pixels). |
| `hull_threshold` | `6` | How far above the noise (robust z-score) counts as "this phone saw heat here". |
| `hull_dilate_px` | `24` | Margin around what each phone saw. |
| `flame_mask_px` | `44` | Radius of the circle masked (and carved out of the hull) around the flame, which is too bright to read. |
| `round_prior` | `true` | Assume a roughly round plume cross-section (see below). |
| `export_vox_mm` | `4` | Voxel size of `field/volume.npy`. |
| `box_center_m` | from the flame | Centre of the 3D box, if there's no flame to find. |

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
| `field/orbit.mp4` | The plume playing while the view circles it: density on the left, the shadowgram a phone there would film on the right. |
| `field/orbit_raw.mp4` | The same without the hull (only with `render.raw_orbit: true`; shows the limits of the camera layout). |
| `field/slices.mp4` | Horizontal cuts through the plume at several heights, seen from above. |
| `field/check_seen_vs_predicted.mp4` | Each phone's shadowgram next to the one the field predicts. The log gives their correlation (signed, and of local energy). |
| `field/training_progress.mp4` | The field's predictions sharpening during training. |
| `field/field.pt` | The trained model; `render` draws its views straight from it. |
| `field/volume.npy` | The 4D field sampled on a grid: (time, x, y, z), `field.export_vox_mm` voxels, float16. |
| `field/fit.pkl` | The prepared shadowgrams, hull and training snapshots (input to `render`). |
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

## Web export

`tools/export_web.py` turns a finished run into small static files for a web page: the rig, the calibration board as each phone saw it, audio sync tracks, cropped clips, training snapshots and the 4D field (uint8, gzipped):
`python tools/export_web.py path/to/project.json path/to/site/public/hotair [--vox 2] [--only field]`. The field is written as one gzipped volume per frame (`field/NNN.gz`) so the page can stream it. `--vox` resamples it from the model (mm per voxel; 2 mm is about as fine as is worth it, finer adds mostly noise), `--only field` skips the videos; brightness is scaled to the 99th percentile with a 0.7 gamma so a hot spot at the flame doesn't wash out the plume.

## How the 3D part works (short version)

A shadowgram is, to first order, the 2D Laplacian of the air's density summed along each line of sight. So each phone's frames are turned into signed shadowgrams (frame ÷ empty sheet − 1, high-passed, divided by the noise level) at full resolution, and a 4D K-Planes field (six 2D feature planes over xy, xz, yz, xt, yt, zt at four scales down to 1.25 mm, multiplied together, then a small network) is fitted so that the Laplacian of its rendered projections matches them. Training renders 32×32 patches of neighbouring rays so the Laplacian can be taken, with smoothness in space and time.

Two things keep the density where the phones say it is:

- **Hull.** Wherever a phone saw nothing above the noise (smoothed over a few frames so it doesn't flicker), its line of sight is kept empty. Nothing is kept below the flame.
- **Round prior.** With the phones on a narrow arc, each height's hull is a long sliver pointing away from them: its width is measured, its depth isn't. The prior keeps density inside a circle around the sliver's centre, sized from its narrow axis. Turn it off with `field.round_prior: false` if your phones surround the subject.

The result is a **shape field** (where the heat is and how it moves), not calibrated temperature. Views far from the phones' arc rely on the round prior.

## Run times (for reference)

On an M4 Pro with three 1080p phones and a 6 s field window: sync 3 s, calibration about 1.5 min (cached afterwards), processing 68 s of footage about 3 min, preparing the field about 1 min, training about 0.2 s per step (12000 steps ≈ 40 min), render about 5 min.

## Troubleshooting

- **"weak match" during sync**: that phone's audio didn't line up clearly. Clap in view next time. You can check with `sync_check.mp4`.
- **A phone reads the board in very few frames**: it was too far from the board, or the screen was too dim. That phone gets placed from the flame instead (`place` step), which works if the other phones are board-calibrated.
- **Halos or streaks in heat.mp4**: the sheet is moving or wrinkled. Tape it flat.
- **3D looks stretched front to back**: the phones are too close together in angle. Spread them wider; `field.round_prior` (on by default) helps in the meantime.
- **Predicted shadowgrams look faint or mushy**: train longer (`field.steps`), or shorten `field_window`. Every pixel of every frame has to be seen many times.
- **Nothing in the 3D box**: the box is placed above the flame automatically. If you're not using a flame, set `field.box_center_m` (metres, in the anchor phone's coordinates; see calibration.json).
