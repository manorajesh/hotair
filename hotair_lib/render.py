"""Videos of the 3D result and of the fitting process. Views are ray-marched straight from the fitted field."""
import cv2
import numpy as np

from .media import Writer


def _fire(x):
    x = np.clip(x, 0, 1)
    r, g, b = np.clip(1.6 * x, 0, 1), np.clip(1.6 * x - 0.45, 0, 1), np.clip(2.2 * x - 1.3, 0, 1) * 0.9 + 0.1 * x
    return (np.stack([b, g, r], -1) * 255).astype(np.uint8)


def _grey(x, contrast):
    g = np.clip(128 + 127 * x / contrast, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _txt(img, s, y, scale=0.7, col=(210, 210, 210), x=24):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def _even(n):
    return n + n % 2


def _look_at(pos, target):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)          # world y points down
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ pos


def orbit(model, scene, data, gain, path, fps, H=720, W=480, ns=96, loops=1, density_lim=None, sg_contrast=None,
          hull=True, title="", log=print):
    """The plume plays `loops` times while a virtual camera circles it once.
    Left: volume rendering of the density. Right: the synthetic shadowgram from that viewpoint (what a phone there would film)."""
    import torch
    from .field import laplacian_of
    scene.use_hull = hull
    T = data["nT"]
    lo, hi = data["lo"], data["hi"]
    c = (lo + hi) / 2
    ref_cam = data["cams"].get("C") or next(iter(data["cams"].values()))
    v0 = -ref_cam["R"].T @ ref_cam["t"] - c
    v0[1] = 0
    D = np.linalg.norm(v0)
    v0 /= D
    size = hi - lo
    f = H * D / (size[1] * 1.08)
    K = np.array([[f, 0, (W + 3) / 2], [0, f, (H + 3) / 2], [0, 0, 1]])
    Kinv = torch.tensor(np.linalg.inv(K), dtype=torch.float32, device=scene.dev)
    vv, uu = np.mgrid[0:H + 4, 0:W + 4]
    pix = torch.tensor(np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], -1), dtype=torch.float32, device=scene.dev)
    dirs_cam = pix @ Kinv.T
    total = T * loops
    gk = float(np.exp(np.mean(np.log([gain(k) for k in data["cams"]]))))
    # Laplacian in this view's pixels -> same units as the real cameras (scale by pixel area)
    f_real = float(np.mean([cm["K"][0, 0] for cm in data["cams"].values()]))
    lap_scale = gk * (f_real / f) ** 2
    w = Writer(path, 2 * W, H, fps)
    stills = []
    step_m = (size[2] * 1.4) / ns
    for n in range(total):
        ti, ang = n % T, 2 * np.pi * n / total
        ca, sa = np.cos(ang), np.sin(ang)
        dirv = np.array([v0[0] * ca + v0[2] * sa, 0, -v0[0] * sa + v0[2] * ca])
        R, t = _look_at(c + D * dirv, c)
        Rt = torch.tensor(R, dtype=torch.float32, device=scene.dev)
        d = dirs_cam @ Rt
        d = d / d.norm(dim=-1, keepdim=True)
        o = torch.tensor(-R.T @ t, dtype=torch.float32, device=scene.dev).expand_as(d)
        dens, proj = [], []
        with torch.no_grad():
            for i in range(0, len(d), 16384):
                oo, dd = o[i:i + 16384], d[i:i + 16384]
                m = len(oo)
                sig, dt = scene.march(model, oo, dd, scene.tnorm(torch.full((m,), ti, device=scene.dev)), ns,
                                      torch.full((m,), 0.5, device=scene.dev))
                proj.append((sig * dt[:, None]).sum(1))
                dens.append(sig * (dt[:, None] / step_m))
            P = torch.cat(proj).view(1, H + 4, W + 4)
            sgv = (laplacian_of(P)[0] * lap_scale).cpu().numpy()
            sig = torch.cat(dens).view(H + 4, W + 4, ns)[2:-2, 2:-2]
            if density_lim is None:
                nz = sig[sig > 0].cpu().numpy()
                density_lim = float(np.percentile(nz, 99.5)) if nz.size else 1.0
            s = (sig / density_lim).clamp(min=0) * 0.12
            img = (s * torch.exp(-0.5 * torch.cumsum(s, -1))).sum(-1).cpu().numpy() / 1.6
        if sg_contrast is None:
            sg_contrast = max(float(np.percentile(np.abs(sgv), 99.7)), 1e-6)
        left = _fire(img ** 0.75)
        right = _grey(sgv, sg_contrast)
        frame = np.hstack([left, right])
        px10 = int(0.10 * f / D)
        cv2.line(frame, (30, H - 40), (30, H - 40 - px10), (200, 200, 200), 2)
        _txt(frame, "10 cm", H - 40 - px10 // 2, 0.55, x=40)
        _txt(frame, f"{title}density   t = {data['t'][ti] - data['t'][0]:.2f} s", 34, 0.6)
        _txt(frame, f"view {np.degrees(ang):3.0f} deg", 60, 0.55, (150, 150, 150))
        _txt(frame, "synthetic shadowgram", 34, 0.6, x=W + 24)
        w.write(frame)
        if n in (0, total // 8, total // 4, 3 * total // 8):
            stills.append(frame)
        if n % 30 == 0:
            log(f"    orbit frame {n}/{total}")
    w.close()
    scene.use_hull = True
    return np.vstack(stills) if stills else None, density_lim, sg_contrast


def slices(vol, data, path, fps, n=6):
    """Horizontal cuts through the plume at increasing heights above the flame (seen from above)."""
    vol = vol.astype(np.float32)
    T, nx, ny, nz = vol.shape
    nzv = vol[vol > 0]
    lim = (np.percentile(nzv, 99.5) if nzv.size else 1.0) + 1e-9
    ys_m = np.linspace(data["lo"][1], data["hi"][1], ny)              # y (down) of each index
    heights = np.linspace(0.03, 0.03 + 0.06 * (n - 1), n)             # 3, 9, 15 ... cm above the flame
    idx = [int(np.argmin(np.abs(ys_m - (data["F"][1] - h)))) for h in heights]
    tile = 260
    w = Writer(path, tile * n, tile + 40, fps)
    for ti in range(T):
        row = []
        for y, hgt in zip(idx, heights):
            sl = vol[ti, :, y, :].T                                     # (z, x)
            im = cv2.resize(_fire((sl / lim) ** 0.8), (tile, tile), interpolation=cv2.INTER_CUBIC)
            im = np.vstack([np.zeros((40, tile, 3), np.uint8), im])
            _txt(im, f"{hgt * 100:.0f} cm above flame", 28, 0.55)
            row.append(im)
        w.write(np.hstack(row))
    w.close()


def _corrs(o, p, v):
    sc = np.corrcoef(cv2.GaussianBlur(o, (0, 0), 1.0)[v], cv2.GaussianBlur(p, (0, 0), 1.0)[v])[0, 1]
    eo = cv2.GaussianBlur(np.abs(o), (0, 0), 4.0)
    ep = cv2.GaussianBlur(np.abs(p), (0, 0), 4.0)
    return sc, np.corrcoef(eo[v], ep[v])[0, 1]


def check(data, preds, path, fps, contrast=4.0, height=640):
    """For every camera: the shadowgram it saw (left) vs the one the 3D field predicts (right), over time.
    Returns per camera the median correlation of the signed shadowgrams and of their local energy."""
    cams = list(preds)
    T = preds[cams[0]].shape[0]
    tiles_w = []
    for k in cams:
        _, Hc, Wc = data["cams"][k]["sg"].shape
        tiles_w.append(int(round(Wc * height / Hc)))
    W = _even(sum(2 * tw + 30 for tw in tiles_w))
    H = _even(height + 44)
    w = Writer(path, W, H, fps)
    corr = {k: ([], []) for k in cams}
    for t in range(T):
        canvas = np.full((H, W, 3), 25, np.uint8)
        x = 0
        for k, tw in zip(cams, tiles_w):
            c = data["cams"][k]
            o, p, v = c["sg"][t].astype(np.float32), preds[k][t], c["valid"][t]
            if v.sum() > 200:
                a, b = _corrs(o, p, v)
                corr[k][0].append(a)
                corr[k][1].append(b)
            im_o = _grey(np.where(v, o, 0), contrast)
            im_o[~v] = (im_o[~v] * 0.35).astype(np.uint8)
            pair = np.hstack([cv2.resize(im_o, (tw, height), interpolation=cv2.INTER_AREA),
                              np.full((height, 4, 3), 25, np.uint8),
                              cv2.resize(_grey(p, contrast), (tw, height), interpolation=cv2.INTER_AREA)])
            canvas[44:44 + height, x:x + pair.shape[1]] = pair
            _txt(canvas, f"{k}: seen | predicted", 30, 0.6, x=x + 8)
            x += pair.shape[1] + 26
        w.write(canvas)
    w.close()
    return {k: (float(np.nanmedian(a)) if a else float("nan"), float(np.nanmedian(b)) if b else float("nan"))
            for k, (a, b) in corr.items()}


def training(snaps, data, path, contrast=4.0, height=420):
    """How the field's predicted shadowgram of each camera sharpened during training (middle frame)."""
    cams = list(snaps[0][1])
    mid = data["nT"] // 2
    tiles = []
    for step, pr in snaps:
        row = []
        for k in cams:
            c = data["cams"][k]
            o, v = c["sg"][mid].astype(np.float32), c["valid"][mid]
            tw = int(round(o.shape[1] * height / o.shape[0]))
            row.append(np.hstack([cv2.resize(_grey(np.where(v, o, 0), contrast), (tw, height), interpolation=cv2.INTER_AREA),
                                  np.full((height, 4, 3), 25, np.uint8),
                                  cv2.resize(_grey(pr[k], contrast), (tw, height), interpolation=cv2.INTER_AREA),
                                  np.full((height, 20, 3), 25, np.uint8)]))
        im = np.vstack([np.full((40, sum(r.shape[1] for r in row), 3), 25, np.uint8), np.hstack(row)])
        _txt(im, f"training step {step}   (each camera: seen | predicted)", 28, 0.6)
        tiles.append(im)
    H, W = _even(tiles[0].shape[0]), _even(tiles[0].shape[1])
    w = Writer(path, W, H, 4)
    for im in tiles + [tiles[-1]] * 8:
        c = np.full((H, W, 3), 25, np.uint8)
        c[:im.shape[0], :im.shape[1]] = im
        w.write(c)
    w.close()
