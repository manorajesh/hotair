# Hot Air: project context for Claude Code

## The project
"Hot Air" is a CMU Experimental Capture project (Golan Levin and Nica Ross). The aim is to capture the invisible warm breath of speech in 3D and over time (4D), and explore how emotion moves through air (whisper, laugh, gasp, "I love you", outburst, two people arguing). The plan has two layers: a live single-camera installation where visitors see their own breath, and an archive of 3D/4D captures.

The author previously built a LiDAR rig (SICK LMS 200, own Python driver, 360 camera and gyro) and scanned Pittsburgh's city steps.

## What we learned (in order)
1. **BOS (background-oriented schlieren)** with a Canon 5D Mk II, a 50 mm lens and a printed speckle board **failed for breath**. A 50 mm lens on 1080p video gives under 0.1 px of speckle shift for breath, which is below the compression noise. BOS needs a long lens (135–200 mm), RAW stills, or a very cold room.
2. **Retroreflective shadowgraph (Edgerton-style)** works: a big retroreflective sheet, with each phone's own flash as the light source (it sits about 1 cm from the lens). A lighter or candle plume shows clearly. **Breath ("haaa") is at the noise floor** with this setup. Things to try for breath: pursed-lip jets, plosives, a cold room (larger ΔT), the breath 3–4 m from the sheet, still air, averaging repeated breaths. Mirror schlieren is the fallback if breath is still too weak.
3. **Affordable thermal cameras (LWIR) can't see breath in open air.** Only mid-wave (CO₂, 4.3 µm) cameras can, and those are expensive.
4. **The 3-phone test worked** (iPhone XS, 1080p60, flash on, AE/AF lock, lighter about 1 m in front of the sheet): synced, calibrated, and reconstructed a 4D "shape field" of the plume.

## The 3-phone test (data in the project folder)
- Files: `stage_left_*`, `center_*`, `stage_right_*` (`*_calib.MOV` with the ChArUco board on a laptop screen, `*_lighter.MOV` for the take). The board is 8×5 squares, 28.36 mm, DICT_5X5_100.
- Sync from audio (no clap in the lighter takes; room noise worked). Take: L = C + 6.9388 s, R = C − 4.2447 s. Calibration clips: L = C + 12.798 s, R = C − 14.604 s.
- Calibration: L and R read the board at about 3.7–4 m and have f ≈ 3290 px (2× lens; centred principal point, no distortion). They are 2.15 m apart with 31° between views. The **centre phone couldn't read the board** (it was too small), so it was placed from the flame (multi-view triangulation, then PnP): about 1.41 m along and 0.65 m behind L. The flame is about 2.7 m from L.
- 3D: a K-Planes 4D field fitted to per-camera heat maps (|high-pass of the frame / reference ratio|) in a 26×45×26 cm box above the flame. Seen-vs-predicted correlation: 0.94 / 0.87 / 0.91 (1500 steps). Raw output has limited-angle "walls"; soft space carving fixes most of it. Depth is the weakest axis because the cameras span only 31°.

## The pipeline (this folder)
`python hotair.py all project.json` runs sync → calibrate → process → place → field → render. Each step can be re-run on its own. See README.md for the config and outputs.
- `hotair_lib/sync.py`: audio cross-correlation, coarse then fine.
- `hotair_lib/calibrate.py`: ChArUco detection on frames sampled at identical reference-clock times; focal-only intrinsics; pairwise stereo on near-still frames with outlier pruning; BFS chaining from the anchor; `place_from_points` for phones without a board fix (uses the board-calibrated phones' focal length when `same_lens` is true).
- `hotair_lib/maps.py`: per-camera clips (side, side_highlight, air, heat, original_synced), flame tracking, field-window data.
- `hotair_lib/field.py`: training data prep (flame and hand masks, sheet-edge erosion), K-Planes model, ray-marched training (device auto: cuda/mps/cpu), export, soft carving.
- `hotair_lib/render.py`: orbit, slices, seen-vs-predicted, training progress.
- `tools/single_camera_bos.py`: single-clip BOS/shadowgraph; `tools/make_charuco.py`: calibration image.

Tested end-to-end on the 3-phone take in a 2-core cloud workspace (CPU only). It has **not yet been run on the M4 Pro**: the first things to check there are the MPS device path and VideoToolbox decoding (`-hwaccel videotoolbox` in `media.py`).

## Next steps
- Next shoot: phones at ±45° (sheet bent to face them), 4K for calibration, board within about 1.5 m of each phone, clap at every take, don't move anything after calibrating. Then try 5 phones.
- Breath: test pursed-lip blows and plosives in a cold room; add repeat-averaging (stack several identical breaths) to the processing.
- Idea: "heat as tracer": a candle or hot drink under the mouth, so speech visibly bends the warm plume. Works with the current sensitivity.
- Possible improvements: physical shadowgraph forward model (true density rather than shape); wider camera arcs; a live single-camera mode for the installation.
