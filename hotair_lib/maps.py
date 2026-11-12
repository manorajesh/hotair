"""Per-camera processing of the take: shadowgraph, heat maps, flame tracking and all the per-camera clips.

For each camera this writes (in out/cameras/<cam>/):
  side.mp4             original | processed air, side by side (whole analysis window)
  side_highlight.mp4   the same, trimmed to the highlight window
  air.mp4              processed shadowgraph only (grey: light/dark = air bending light)
  heat.mp4             heat magnitude (colour), the signal the 3D step uses
  original_synced.mp4  untouched footage, cut to the shared reference clock (for sync checks)
  strongest.png        the frame with the most visible heat
  reference.png        the empty-sheet reference frame
  maps.npz             flame track in full-resolution pixels (used to place the 3D box and mask the flame)
"""
import os

import cv2
import numpy as np

from .media import Writer, decode, probe


def masked_highpass(r, valid, sigma):
    v = valid.astype(np.float32)
    bl = cv2.GaussianBlur(r * v, (0, 0), sigma)
    bw = cv2.GaussianBlur(v, (0, 0), sigma) + 1e-4
    return (r - bl / bw) * v


def find_flame(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] < 30) & (hsv[..., 1] > 120) & (hsv[..., 2] > 180)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if not 4 <= st[k, cv2.CC_STAT_AREA] < 20000:
        return None
    ys, xs = np.where(lab == k)
    w = hsv[ys, xs, 2].astype(float)
    return np.array([np.average(xs, weights=w), np.average(ys, weights=w)])


def label(img, text):
    cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def process_camera(cam, path, offset, cfg, outdir, log=print):
    tk = cfg["take"]
    pr = cfg.get("processing", {})
    W = pr.get("clip_width", 960)
    fps = pr.get("clip_fps", 30)
    contrast = pr.get("air_contrast", 0.04)
    heat_contrast = pr.get("heat_contrast", 0.02)
    info = probe(path)
    s = W / info["width"]
    os.makedirs(outdir, exist_ok=True)

    r0, r1 = tk["reference"]
    ref = np.median(np.stack([g.astype(np.float32) for g in decode(path, r0 + offset, r1 - r0, 30, W)]), axis=0)
    sheet = ref > 0.45 * np.percentile(ref, 99)
    edge = max(3, int(round(7 * W / 480)))
    sheet_core = cv2.erode(sheet.astype(np.uint8), np.ones((edge, edge), np.uint8)) > 0
    cv2.imwrite(os.path.join(outdir, "reference.png"), ref.astype(np.uint8))

    a0, a1 = tk["analysis"]
    h0, h1 = tk.get("highlight", [a0, a1])
    H = ref.shape[0]
    wr = dict(side=Writer(os.path.join(outdir, "side.mp4"), 2 * W, H, fps),
              hl=Writer(os.path.join(outdir, "side_highlight.mp4"), 2 * W, H, fps),
              air=Writer(os.path.join(outdir, "air.mp4"), W, H, fps, color=False),
              heat=Writer(os.path.join(outdir, "heat.mp4"), W, H, fps),
              orig=Writer(os.path.join(outdir, "original_synced.mp4"), W, H, fps))
    track = {}
    best, best_score = None, -1
    for i, bgr in enumerate(decode(path, a0 + offset, a1 - a0, fps, W, color=True)):
        t = a0 + i / fps
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        valid = sheet & (g > 0.45 * ref)
        r = np.where(valid, (g + 1) / (ref + 1) - 1, 0)
        core = cv2.erode(valid.astype(np.uint8), np.ones((edge, edge), np.uint8)) > 0
        core &= sheet_core
        air = np.clip(128 + 127 * r / contrast, 0, 255).astype(np.uint8)
        air[~valid] = 0                                   # you and anything off the sheet: black silhouette
        hp = masked_highpass(r, core, 6 * W / 480)
        sig = 1.5 * W / 480
        mag = cv2.GaussianBlur(np.abs(hp), (0, 0), sig) / (cv2.GaussianBlur(core.astype(np.float32), (0, 0), sig) + 1e-4)
        mag[~core] = 0
        heat = cv2.applyColorMap(np.clip(255 * mag / heat_contrast, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat[~core] = 0
        fl = find_flame(bgr)
        if fl is not None:
            track[round(t, 3)] = (fl / s).tolist()          # full-resolution pixels
        side = np.hstack([label(bgr.copy(), f"{cam}  original"), label(cv2.cvtColor(air, cv2.COLOR_GRAY2BGR), f"{cam}  air  t={t:.2f}s")])
        wr["side"].write(side)
        if h0 <= t < h1:
            wr["hl"].write(side)
        wr["air"].write(air)
        wr["heat"].write(heat)
        wr["orig"].write(bgr)
        sc = float(np.percentile(mag[core], 99.5)) if core.any() else 0
        if sc > best_score:
            best_score, best = sc, side
    for w in wr.values():
        w.close()
    if best is not None:
        cv2.imwrite(os.path.join(outdir, "strongest.png"), best)
    np.savez_compressed(os.path.join(outdir, "maps.npz"), track_t=np.array(list(track.keys())), track_uv=np.array(list(track.values())))
    log(f"  {cam}: {i + 1} frames processed, flame seen in {len(track)}")
    return track
