#!/usr/bin/env python3
"""Export a finished run as small static files for the web page (mano.raje.sh/hot-air).

  python tools/export_web.py local/project.json ~/code/mano.raje.sh/public/hotair

Writes:
  rig.json            phone positions and view cones, flame, 3D box (metres, y up, flame at the origin)
  sync.json           audio envelopes of every phone around a stretch of the take, and the offsets found
  clip_<cam>.mp4      cropped take: original | shadowgram side by side, around the field window
  train_<cam>.webp    what the phone saw (first tile), then the field's prediction at each training snapshot
  train.json          snapshot steps, loss curve, tile sizes
  field.bin.gz        the 3D field: uint8 frames, x fastest, then y, then z
  field.json          its shape, box, frame rate
"""
import gzip
import json
import os
import pickle
import re
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hotair as H  # noqa: E402
from hotair_lib import calibrate as C  # noqa: E402
from hotair_lib.media import FFMPEG, audio, decode  # noqa: E402
from hotair_lib.sync import SR, _env  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0])   # camera-style (y down, z forward) -> y up


def grey(x, contrast):
    return np.clip(128 + 127 * x / contrast, 0, 255).astype(np.uint8)


def rig(cfg, calib, data, dst):
    F = data["F"]
    cams = {}
    for k, c in calib["cameras"].items():
        if "R" not in c:
            continue
        R, t, K = np.array(c["R"]), np.array(c["t"]), np.array(c["K"])
        pos = FLIP @ (-R.T @ t - F)
        cams[k] = dict(pos=pos.round(4).tolist(), right=(FLIP @ R[0]).round(5).tolist(), up=(FLIP @ -R[1]).round(5).tolist(),
                       forward=(FLIP @ R[2]).round(5).tolist(), f=float(K[0, 0]), w=1920, h=1080)
    lo, hi = FLIP @ (data["lo"] - F), FLIP @ (data["hi"] - F)
    json.dump(dict(cameras=cams, box=dict(min=np.minimum(lo, hi).round(4).tolist(), max=np.maximum(lo, hi).round(4).tolist())),
              open(os.path.join(dst, "rig.json"), "w"))


def sync(cfg, offsets, dst, span=16.0, margin=3.0):
    rate = 100
    env = {}
    for k, cam in cfg["cameras"].items():
        e, r = _env(audio(cam["take"], SR), SR // rate)
        env[k] = e
    ref = cfg["reference_camera"]
    # the stretch of the reference clock (inside every phone's recording) with the most going on
    t_lo = max(-offsets[k] for k in env) + margin
    t_hi = min(len(env[k]) / rate - offsets[k] for k in env) - span - margin
    best, best_s = t_lo, -1
    for s in np.arange(t_lo, t_hi, 0.5):
        i = int(s * rate)
        v = np.clip(env[ref][i:i + int(span * rate)], 0, None)
        sc = float((v ** 2).sum())
        if sc > best_s:
            best, best_s = s, sc
    tracks = {}
    for k in env:
        i0 = int((best - margin + offsets[k]) * rate)
        seg = env[k][i0:i0 + int((span + 2 * margin) * rate)]
        tracks[k] = np.clip(seg, -1, 6).round(2).tolist()
    json.dump(dict(rate=rate, margin=margin, span=span, start=round(best, 2), reference=ref,
                   offsets={k: round(v, 4) for k, v in offsets.items()}, tracks=tracks),
              open(os.path.join(dst, "sync.json"), "w"))


def clips(cfg, offsets, data, dst, pad=2.0, fps=30, contrast=0.04):
    f0, f1 = cfg["take"]["field_window"]
    r0, r1 = cfg["take"]["reference"]
    out = {}
    for k, c in data["cams"].items():
        path, off = cfg["cameras"][k]["take"], offsets[k]
        _, Hc, Wc = c["sg"].shape
        x0, y0 = c["x0"], c["y0"]
        Wc -= Wc % 2
        Hc -= Hc % 2
        ref = np.median(np.stack([g.astype(np.float32) for g in decode(path, r0 + off, r1 - r0, 15)]), 0)[y0:y0 + Hc, x0:x0 + Wc]
        sheet = ref > 0.45 * np.percentile(ref, 99)
        p = os.path.join(dst, f"clip_{k}.mp4")
        enc = subprocess.Popen([FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * Wc}x{Hc}", "-r", str(fps),
                                "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-g", "10", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", p], stdin=subprocess.PIPE)
        n = 0
        for bgr in decode(path, f0 - pad + off, f1 - f0 + 2 * pad, fps, color=True):
            b = bgr[y0:y0 + Hc, x0:x0 + Wc]
            g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
            valid = sheet & (g > 0.45 * ref)
            a = grey(np.where(valid, (g + 1) / (ref + 1) - 1, 0), contrast)
            a[~valid] = 0
            enc.stdin.write(np.hstack([b, cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)]).tobytes())
            n += 1
        enc.stdin.close()
        enc.wait()
        out[k] = dict(w=Wc, h=Hc, frames=n, fps=fps, start=f0 - pad)
    return out


def _fit(img, w, h, fill):
    """Scale to fit inside w x h, centred, padded with `fill`, so every phone's tiles share one shape."""
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w), fill, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def training(cfg, fit, dst, height=320, aspect=0.72, contrast=4.0):
    data, snaps = fit["data"], fit["snaps"]
    mid = data["nT"] // 2
    tiles = {}
    w = int(round(height * aspect))
    for k, c in data["cams"].items():
        o, v = c["sg"][mid].astype(np.float32), c["valid"][mid]
        seen = grey(np.where(v, o, 0), contrast)
        seen[~v] = (seen[~v] * 0.3).astype(np.uint8)
        rows = [_fit(seen, w, height, 38)] + [_fit(grey(pr[k], contrast), w, height, 128) for _, pr in snaps]
        strip = np.vstack(rows)
        cv2.imwrite(os.path.join(dst, f"train_{k}.webp"), strip, [cv2.IMWRITE_WEBP_QUALITY, 82])
        tiles[k] = dict(w=w, h=height)
    log = open(H.out(cfg, "log.txt")).read()
    log = log[log.rfind("FIELD: training"):]
    loss = [[int(a), float(b)] for a, b in re.findall(r"step\s+(\d+)\s+fit\s+([\d.]+)", log)]
    json.dump(dict(steps=[s for s, _ in snaps], loss=loss, tiles=tiles), open(os.path.join(dst, "train.json"), "w"))


def field(cfg, data, dst, every=2):
    vol = np.load(H.out(cfg, "field", "volume.npy"))[::every].astype(np.float32)
    nz = vol[vol > 0]
    lim = float(np.percentile(nz, 99.7)) if nz.size else 1.0
    q = np.clip(vol / lim * 255, 0, 255).astype(np.uint8)          # (t, x, y, z) in camera-style axes
    q = q[:, :, ::-1, ::-1]                                           # flip y and z -> y up
    q = np.ascontiguousarray(q.transpose(0, 3, 2, 1))                 # (t, z, y, x): x fastest
    with gzip.open(os.path.join(dst, "field.bin.gz"), "wb", compresslevel=9) as f:
        f.write(q.tobytes())
    lo, hi = FLIP @ (data["lo"] - data["F"]), FLIP @ (data["hi"] - data["F"])
    fps = data.get("fps", 30) / every
    json.dump(dict(nx=q.shape[3], ny=q.shape[2], nz=q.shape[1], frames=q.shape[0], fps=fps,
                   min=np.minimum(lo, hi).round(4).tolist(), max=np.maximum(lo, hi).round(4).tolist()),
              open(os.path.join(dst, "field.json"), "w"))
    return q.shape


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cfg = H.load_cfg(sys.argv[1])
    dst = os.path.expanduser(sys.argv[2])
    os.makedirs(dst, exist_ok=True)
    calib = C.load(H.out(cfg, "calibration.json"))
    offsets = json.load(open(H.out(cfg, "sync.json")))["take"]
    fit = pickle.load(open(H.out(cfg, "field", "fit.pkl"), "rb"))
    data = fit["data"]
    rig(cfg, calib, data, dst)
    print("rig.json")
    sync(cfg, offsets, dst)
    print("sync.json")
    training(cfg, fit, dst)
    print("train_*.webp, train.json")
    print("field", field(cfg, data, dst))
    meta = clips(cfg, offsets, data, dst)
    json.dump(meta, open(os.path.join(dst, "clips.json"), "w"))
    print("clips", meta)


if __name__ == "__main__":
    main()
