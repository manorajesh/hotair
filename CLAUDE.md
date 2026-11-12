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
- Calibration: L and R read the board at about 3.7–4 m and have f ≈ 3290 px (2× lens; centred principal point, no distortion). They are 2.15 m apart with 31° between views. The centre phone's board was too small to read directly; retrying with enlarged, contrast-boosted frames found it in 185 frames (f ≈ 3131 px, about 1.22 m along and 0.43 m behind L). Earlier it was placed from the flame instead (about 1.41 m along, 0.65 m behind). The flame is about 2.7 m from L.
- 3D, first version: a K-Planes 4D field fitted to blurred per-camera heat maps (|high-pass|) at 480 px wide. Correlation 0.94 / 0.87 / 0.91, but a blurry blob by construction.
- 3D, now: fitted to signed full-res shadowgrams through a shadowgraph forward model, 30 fps, in a 26×45×26 cm box above the flame. It needs many steps (each pixel of each frame must be seen often): a 1 s window at 2000 steps reached signed correlation 0.56–0.66. Depth is the weakest axis because the cameras span only 31°; a hull plus a round cross-section prior keep the plume from smearing through the box.

## The pipeline (this folder)
`python hotair.py all project.json` runs sync → calibrate → process → place → field → render. Each step can be re-run on its own. See README.md for the config and outputs.
- `hotair_lib/sync.py`: audio cross-correlation, coarse then fine.
- `hotair_lib/calibrate.py`: ChArUco detection on frames sampled at identical reference-clock times; focal-only intrinsics; pairwise stereo on near-still frames with outlier pruning; BFS chaining from the anchor; `place_from_points` for phones without a board fix (uses the board-calibrated phones' focal length when `same_lens` is true).
- `hotair_lib/maps.py`: per-camera clips (side, side_highlight, air, heat, original_synced), flame tracking.
- `hotair_lib/field.py`: full-res signed shadowgrams straight from the takes (flame and hand masks, sheet-edge erosion), per-camera hull, round cross-section prior, multiscale K-Planes, shadowgraph forward model (Laplacian of rendered projections on 32×32 ray patches), volume export. Device auto: cuda/mps/cpu.
- `hotair_lib/render.py`: orbit ray-marched from the model (density + synthetic shadowgram), slices, signed seen-vs-predicted, training progress.
- `tools/export_web.py`: static files for the web page (mano.raje.sh/hot-air).
- `tools/single_camera_bos.py`: single-clip BOS/shadowgraph; `tools/make_charuco.py`: calibration image.

Runs end-to-end on the M4 Pro (MPS, VideoToolbox decoding). Homebrew's ffmpeg has no `drawtext`, so grid videos go unlabelled there. On MPS, keep tensor shapes repeating (see the bucketed hull filter in `Scene.march`): new shapes compile new kernels and training slows down step by step.

## Next steps
- Next shoot: phones at ±45° (sheet bent to face them), 4K for calibration, board within about 1.5 m of each phone, clap at every take, don't move anything after calibrating. Then try 5 phones.
- Breath: test pursed-lip blows and plosives in a cold room; add repeat-averaging (stack several identical breaths) to the processing.
- Idea: "heat as tracer": a candle or hot drink under the mouth, so speech visibly bends the warm plume. Works with the current sensitivity.
- Possible improvements: wider camera arcs (then drop the round prior); calibrated density from the shadowgraph gain; a live single-camera mode for the installation.
