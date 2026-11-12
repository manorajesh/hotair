#!/usr/bin/env python3
"""Hot Air: multi-phone retroreflective shadowgraph -> synced clips -> 4D neural field of the air.

  python hotair.py all       project.json     # everything, in order
  python hotair.py sync      project.json     # 1. line the cameras up from their audio
  python hotair.py calibrate project.json     # 2. camera positions from the ChArUco clips
  python hotair.py process   project.json     # 3. per-camera clips + heat maps + flame tracks
  python hotair.py place     project.json     # 4. place phones that never read the board, from the flame
  python hotair.py field     project.json     # 5. fit the 4D neural field
  python hotair.py render    project.json     # 6. 3D videos and checks

Each step saves its results in the output folder, so you can re-run any later step on its own.
See README.md.
"""
import json
import os
import sys
import time

import numpy as np

from hotair_lib import calibrate as C
from hotair_lib import field as Fd
from hotair_lib import maps as Mp
from hotair_lib import render as Rd
from hotair_lib.media import grid_video, probe
from hotair_lib.sync import sync_files


def load_cfg(path):
    cfg = json.load(open(path))
    base = os.path.dirname(os.path.abspath(path))
    for c in cfg["cameras"].values():
        for key in ("take", "calib"):
            if c.get(key):
                c[key] = os.path.join(base, c[key])
    cfg["output"] = os.path.join(base, cfg.get("output", "out"))
    os.makedirs(cfg["output"], exist_ok=True)
    return cfg


def out(cfg, *p):
    return os.path.join(cfg["output"], *p)


def log_to(cfg):
    f = open(out(cfg, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
    return log


def step_sync(cfg, log):
    ref = cfg["reference_camera"]
    log("SYNC: take")
    take = sync_files({k: c["take"] for k, c in cfg["cameras"].items()}, ref, log)
    res = {"take": {k: v[0] for k, v in take.items()}, "take_strength": {k: v[1] for k, v in take.items()}}
    if all(c.get("calib") for c in cfg["cameras"].values()):
        log("SYNC: calibration")
        cal = sync_files({k: c["calib"] for k, c in cfg["cameras"].items()}, ref, log)
        res["calib"] = {k: v[0] for k, v in cal.items()}
    json.dump(res, open(out(cfg, "sync.json"), "w"), indent=1)
    # a quick synced check clip of the raw footage
    a0, a1 = cfg["take"]["analysis"]
    clips = []
    os.makedirs(out(cfg, "sync_check"), exist_ok=True)
    import subprocess
    from hotair_lib.media import FFMPEG
    dur = min(10.0, a1 - a0)
    for k, c in cfg["cameras"].items():
        p = out(cfg, "sync_check", f"{k}.mp4")
        subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{a0 + res['take'][k]:.3f}", "-i", c["take"], "-t", str(dur),
                        "-vf", "scale=640:-2,fps=30", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", p], check=True)
        clips.append(p)
    grid_video(clips, out(cfg, "sync_check.mp4"), labels=list(cfg["cameras"]))
    log("  wrote sync_check.mp4 (first 10 s of the analysis window, all cameras side by side)")


def step_calibrate(cfg, log):
    sync = json.load(open(out(cfg, "sync.json")))
    log("CALIBRATE")
    calib, dets = C.calibrate(cfg, sync["calib"], log, cache=out(cfg, "calibration_detections.pkl"))
    C.save(calib, out(cfg, "calibration.json"))
    _detection_sheet(cfg, sync["calib"], dets, log)
    C.summary(calib, log)


def _detection_sheet(cfg, offsets, dets, log):
    """One image per camera with the detected board corners drawn on a representative frame."""
    import cv2
    from hotair_lib.media import decode
    tiles = []
    for k, c in cfg["cameras"].items():
        d = dets.get(k, [])
        if d:
            t, corners, _ = max(d, key=lambda x: len(x[2]))
            img = next(decode(c["calib"], t + offsets[k], 0.05, 30, color=True)).copy()
            for u, v in corners.astype(int):
                cv2.circle(img, (int(u), int(v)), 6, (0, 255, 0), 2)
        else:
            mid = probe(c["calib"])["duration"] / 2
            img = next(decode(c["calib"], mid, 0.05, 30, color=True)).copy()
        cv2.putText(img, f"{k}: board in {len(d)} frames", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
        tiles.append(cv2.resize(img, (640, 360)))
    cv2.imwrite(out(cfg, "calibration_detections.png"), np.hstack(tiles))
    log("  wrote calibration_detections.png")


def step_process(cfg, log):
    sync = json.load(open(out(cfg, "sync.json")))
    log("PROCESS: per-camera clips and heat maps")
    tracks, dirs = {}, {}
    for k, c in cfg["cameras"].items():
        dirs[k] = out(cfg, "cameras", k)
        tracks[k] = Mp.process_camera(k, c["take"], sync["take"][k], cfg, dirs[k], log)
    names = list(cfg["cameras"])
    for kind in ("air", "heat", "original_synced"):
        grid_video([os.path.join(dirs[k], f"{kind}.mp4") for k in names], out(cfg, f"all_cameras_{kind}.mp4"), labels=names)
    grid_video([os.path.join(dirs[k], "side_highlight.mp4") for k in names], out(cfg, "all_cameras_side_highlight.mp4"),
               cols=1, tile_w=1280)
    log("  wrote all_cameras_air.mp4, all_cameras_heat.mp4, all_cameras_side_highlight.mp4, all_cameras_original_synced.mp4")


def step_place(cfg, log):
    """Place phones that never read the board, using the flame seen by everyone in the take."""
    calib = C.load(out(cfg, "calibration.json"))
    if all("R" in c for c in calib["cameras"].values()):
        log("PLACE: every camera already placed by the board")
        return
    log("PLACE: cameras without a board fix, from the flame")
    tracks = {}
    for k in cfg["cameras"]:
        z = np.load(out(cfg, "cameras", k, "maps.npz"))
        tracks[k] = {round(float(t), 3): np.array(uv) for t, uv in zip(z["track_t"], z["track_uv"])}
    calib = C.place_from_points(calib, tracks, log, same_lens=cfg.get("calibration", {}).get("same_lens", True))
    C.save(calib, out(cfg, "calibration.json"))
    C.summary(calib, log)


def step_field(cfg, log):
    import pickle
    calib = C.load(out(cfg, "calibration.json"))
    sync = json.load(open(out(cfg, "sync.json")))
    dirs = {k: out(cfg, "cameras", k) for k in cfg["cameras"]}
    log("FIELD: preparing full-resolution shadowgrams")
    data = Fd.prepare(cfg, calib, sync["take"], dirs, log)
    os.makedirs(out(cfg, "field"), exist_ok=True)
    log("FIELD: training")
    model, scene, snaps = Fd.train(data, cfg, out(cfg, "field"), log)
    pickle.dump(dict(data=data, snaps=snaps), open(out(cfg, "field", "fit.pkl"), "wb"))
    vox = cfg.get("field", {}).get("export_vox_mm", 4.0)
    vol = Fd.export_volume(model, scene, data, vox)
    np.save(out(cfg, "field", "volume.npy"), vol)
    log(f"  saved field/field.pt (the model) and field/volume.npy {vol.shape} (time, x, y, z; {vox} mm voxels)")


def step_render(cfg, log):
    import pickle
    import cv2
    fit = pickle.load(open(out(cfg, "field", "fit.pkl"), "rb"))
    data, snaps = fit["data"], fit["snaps"]
    fps = data.get("fps", cfg.get("processing", {}).get("field_fps", 30))
    rc = cfg.get("render", {})
    log("RENDER")
    model, scene, gain = Fd.load(out(cfg, "field", "field.pt"), data)
    vol = np.load(out(cfg, "field", "volume.npy"))
    nz = vol[vol > 0].astype(np.float32)
    lim = float(np.percentile(nz, 99.5)) if nz.size else 1.0
    log("  predicting every camera's shadowgram from the field")
    preds = {k: np.stack([Fd.predict_camera(model, scene, gain, k, ti) for ti in range(data["nT"])]) for k in data["cams"]}
    corr = Rd.check(data, preds, out(cfg, "field", "check_seen_vs_predicted.mp4"), fps)
    for k, (a, b) in corr.items():
        log(f"  {k}: seen vs predicted correlation {a:.2f} (signed shadowgram), {b:.2f} (local energy)")
    Rd.training(snaps, data, out(cfg, "field", "training_progress.mp4"))
    Rd.slices(vol, data, out(cfg, "field", "slices.mp4"), fps)
    H, W = rc.get("orbit_size", [720, 480])
    still, lim, sgc = Rd.orbit(model, scene, data, gain, out(cfg, "field", "orbit.mp4"), fps, H=H, W=W,
                               ns=rc.get("samples_per_ray", 96), loops=rc.get("loops", 1), density_lim=lim, log=log)
    cv2.imwrite(out(cfg, "field", "orbit_stills.png"), still)
    if rc.get("raw_orbit", False):
        Rd.orbit(model, scene, data, gain, out(cfg, "field", "orbit_raw.mp4"), fps, H=H, W=W, ns=rc.get("samples_per_ray", 96),
                 loops=rc.get("loops", 1), density_lim=lim, sg_contrast=sgc, hull=False, title="no hull  ", log=log)
    log("  wrote field/check_seen_vs_predicted.mp4, training_progress.mp4, slices.mp4, orbit.mp4, orbit_stills.png")


STEPS = {"sync": step_sync, "calibrate": step_calibrate, "process": step_process, "place": step_place,
         "field": step_field, "render": step_render}


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in list(STEPS) + ["all"]:
        print(__doc__)
        sys.exit(1)
    cfg = load_cfg(sys.argv[2])
    log = log_to(cfg)
    todo = list(STEPS) if sys.argv[1] == "all" else [sys.argv[1]]
    for s in todo:
        t = time.time()
        STEPS[s](cfg, log)
        log(f"  ({s} took {time.time() - t:.0f} s)\n")


if __name__ == "__main__":
    main()
