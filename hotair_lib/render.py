"""Videos of the 3D result and of the fitting process."""
import cv2
import numpy as np
from scipy import ndimage as ndi

from .media import Writer


def _fire(x):
    x = np.clip(x, 0, 1)
    r, g, b = np.clip(1.6 * x, 0, 1), np.clip(1.6 * x - 0.45, 0, 1), np.clip(2.2 * x - 1.3, 0, 1) * 0.9 + 0.1 * x
    return (np.stack([b, g, r], -1) * 255).astype(np.uint8)


def _txt(img, s, y, scale=0.7, col=(210, 210, 210)):
    cv2.putText(img, s, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def orbit(vol, data, path, fps, loops=2, H=900, W=720, title=""):
    """Plume plays `loops` times while the view turns once around it."""
    vol = vol.astype(np.float32)
    T = vol.shape[0]
    size = data["hi"] - data["lo"]
    vox = 0.005
    zoom = tuple(size[i] / vox / vol.shape[i + 1] for i in range(3))
    lim = np.percentile(vol, 99.7) + 1e-9
    w = Writer(path, W, H, fps)
    stills = []
    total = T * loops
    for n in range(total):
        ti, ang = n % T, 360.0 * n / total
        v = ndi.rotate(ndi.zoom(vol[ti], zoom, order=1), ang, axes=(0, 2), reshape=False, order=1)
        s = np.clip(v / lim, 0, None) * 0.15
        img = (s * np.exp(-np.cumsum(s, 2) * 0.6)).sum(2).T / 2.2
        im = _fire(img ** 0.8)
        im = cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_CUBIC)[:, :W]
        canvas = np.zeros((H, W, 3), np.uint8)
        x0 = (W - im.shape[1]) // 2
        canvas[:, x0:x0 + im.shape[1]] = im
        px10 = int(0.10 / size[1] * H)
        cv2.line(canvas, (30, H - 40), (30, H - 40 - px10), (200, 200, 200), 2)
        _txt(canvas, "10 cm", H - 40 - px10 // 2, 0.6)
        _txt(canvas, f"{title}  t = {data['t'][ti] - data['t'][0]:.2f} s", 40)
        _txt(canvas, f"view {ang:3.0f} deg", 70, 0.6, (150, 150, 150))
        w.write(canvas)
        if n in (0, total // 8, total // 4, 3 * total // 8):
            stills.append(canvas)
    w.close()
    return np.hstack(stills)


def slices(vol, data, path, fps, n=6):
    """Horizontal cuts through the plume at increasing heights above the flame (seen from above)."""
    vol = vol.astype(np.float32)
    T, nx, ny, nz = vol.shape
    nzv = vol[vol > 0]
    lim = (np.percentile(nzv, 99) if nzv.size else 1.0) + 1e-9
    ys_m = np.linspace(data["lo"][1], data["hi"][1], ny)              # y (down) of each index
    heights = np.linspace(0.03, 0.03 + 0.06 * (n - 1), n)             # 3, 9, 15 ... cm above the flame
    idx = [int(np.argmin(np.abs(ys_m - (data["F"][1] - h)))) for h in heights]
    tile = 220
    w = Writer(path, tile * n, tile + 40, fps)
    for ti in range(T):
        row = []
        for y, hgt in zip(idx, heights):
            sl = vol[ti, :, max(y - 1, 0):y + 2, :].mean(1).T              # 3-voxel thick cut, (z, x)
            im = cv2.resize(_fire((sl / lim) ** 0.8), (tile, tile), interpolation=cv2.INTER_CUBIC)
            im = np.vstack([np.zeros((40, tile, 3), np.uint8), im])
            _txt(im, f"{hgt * 100:.0f} cm above flame", 28, 0.55)
            row.append(im)
        w.write(np.hstack(row))
    w.close()


def check(data, preds, path, fps, scale=0.015, zoom=3):
    """For every camera: what it saw (left) vs what the 3D field predicts it should see (right), over time."""
    cams = list(preds)
    T = preds[cams[0]].shape[0]
    rows_h = [data["cams"][k]["maps"].shape[1] * zoom for k in cams]
    cols_w = [2 * data["cams"][k]["maps"].shape[2] * zoom + 30 for k in cams]
    W, H = sum(cols_w), max(rows_h) + 40
    W += W % 2
    H += H % 2
    w = Writer(path, W, H, fps)
    corr = {k: [] for k in cams}
    for t in range(T):
        canvas = np.full((H, W, 3), 25, np.uint8)
        x = 0
        for k in cams:
            o, p, v = data["cams"][k]["maps"][t], preds[k][t], data["cams"][k]["valid"][t]
            if v.sum() > 50:
                corr[k].append(np.corrcoef(cv2.GaussianBlur(o, (0, 0), 1.5)[v], cv2.GaussianBlur(p, (0, 0), 1.5)[v])[0, 1])
            pair = np.hstack([o, np.zeros((o.shape[0], 2), np.float32), p])
            im = cv2.resize(_fire(pair / scale), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            canvas[40:40 + im.shape[0], x:x + im.shape[1]] = im
            cv2.putText(canvas, f"{k}: seen | predicted", (x + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
            x += im.shape[1] + 30
        w.write(canvas)
    w.close()
    return {k: float(np.median(v)) if v else float("nan") for k, v in corr.items()}


def training(snaps, data, path, scale=0.015, zoom=3):
    """How the field's predicted view of each camera sharpened during training (middle frame)."""
    cams = list(snaps[0][1])
    mid = data["nT"] // 2
    tiles = []
    for step, pr in snaps:
        row = []
        for k in cams:
            o = data["cams"][k]["maps"][mid]
            pair = np.hstack([o, np.zeros((o.shape[0], 2), np.float32), pr[k]])
            row.append(cv2.resize(_fire(pair / scale), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST))
        h = max(r.shape[0] for r in row)
        row = [cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 20, cv2.BORDER_CONSTANT, value=(25, 25, 25)) for r in row]
        im = np.vstack([np.full((40, sum(r.shape[1] for r in row), 3), 25, np.uint8), np.hstack(row)])
        _txt(im, f"training step {step}   (each camera: seen | predicted)", 28, 0.6)
        tiles.append(im)
    H, W = tiles[0].shape[:2]
    W += W % 2
    H += H % 2
    w = Writer(path, W, H, 4)
    for im in tiles:
        c = np.full((H, W, 3), 25, np.uint8)
        c[:im.shape[0], :im.shape[1]] = im
        w.write(c)
    for _ in range(8):
        w.write(c)
    w.close()
