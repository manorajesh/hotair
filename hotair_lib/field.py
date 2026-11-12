"""4D neural field of the heat: (x, y, z, t) -> density, fitted so that its line integrals match every camera's heat map.

Model: multiscale K-Planes (six 2D feature planes per scale: xy xz yz xt yt zt), features multiplied together,
a small MLP, softplus output. Trained with a differentiable ray-marching renderer.
"""
import os
import time

import cv2
import numpy as np


# ---------------------------------------------------------------- data preparation
def flame_center(calib, tracks, t0, t1):
    from .calibrate import triangulate
    cams = [k for k, c in calib["cameras"].items() if "R" in c and k in tracks]
    P = {k: np.array(calib["cameras"][k]["K"]) @ np.c_[np.array(calib["cameras"][k]["R"]), np.array(calib["cameras"][k]["t"])] for k in cams}
    ts = sorted(set.intersection(*[set(t for t in tracks[k] if t0 <= t < t1) for k in cams]))
    X = [triangulate([tracks[k][t] for k in cams], [P[k] for k in cams]) for t in ts]
    if not X:
        raise RuntimeError("The flame wasn't seen by all calibrated cameras in the field window; set field.box_center_m by hand.")
    return np.median(np.array(X), 0)


def prepare(cfg, calib, camdirs, log=print):
    fc = cfg.get("field", {})
    tracks = {}
    M = {}
    for k, d in camdirs.items():
        z = np.load(os.path.join(d, "maps.npz"))
        M[k] = z
        tracks[k] = {round(float(t), 3): np.array(uv) for t, uv in zip(z["track_t"], z["track_uv"])}
    f0, f1 = cfg["take"]["field_window"]
    F = np.array(fc["box_center_m"]) if fc.get("box_center_m") else flame_center(calib, tracks, f0, f1)
    w, h, d = fc.get("box_size_m", [0.26, 0.45, 0.26])
    below = fc.get("box_below_flame_m", 0.03)
    lo = F + np.array([-w / 2, -(h - below), -d / 2])        # y points down: the plume rises towards -y
    hi = F + np.array([w / 2, below, d / 2])
    log(f"  flame at {np.round(F, 3)} m; 3D box {w * 100:.0f} x {h * 100:.0f} x {d * 100:.0f} cm")
    corners = np.array([[x, y, zz] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for zz in (lo[2], hi[2])])
    out = {}
    nT = None
    for k, z in M.items():
        c = calib["cameras"].get(k, {})
        if "R" not in c:
            log(f"  {k}: not calibrated, skipped")
            continue
        s = float(z["scale"])
        K = np.array(c["K"], float).copy()
        K[:2, :2] *= s
        K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
        Mx = np.eye(4)
        Mx[:3, :3], Mx[:3, 3] = np.array(c["R"]), np.array(c["t"])
        r, valid, ts, ref = z["r"].astype(np.float32), z["valid"], z["t"], z["ref"]
        Hh, Ww = ref.shape
        pc = K @ (Mx[:3, :3] @ corners.T + Mx[:3, 3:])
        pc = (pc[:2] / pc[2]).T
        x0, y0 = np.maximum(np.floor(pc.min(0)).astype(int) - 2, 0)
        x1, y1 = np.minimum(np.ceil(pc.max(0)).astype(int) + 2, [Ww, Hh])
        pf = K @ (Mx[:3, :3] @ F + Mx[:3, 3])
        pf = pf[:2] / pf[2]
        sheet = (ref > 0.45 * np.percentile(ref, 99)).astype(np.uint8)
        inner = cv2.erode(sheet, np.ones((13, 13), np.uint8)) > 0
        tk = tracks[k]
        tt = np.array(sorted(tk)) if tk else np.array([])
        mp, vl = [], []
        rad = fc.get("flame_mask_px", 7)
        for i in range(len(ts)):
            v = (valid[i] & inner).astype(np.float32)
            fu = pf
            if len(tt):
                j = int(np.argmin(np.abs(tt - ts[i])))
                if abs(tt[j] - ts[i]) < 0.2:
                    fu = tk[tt[j]] * s
            cv2.circle(v, (int(fu[0]), int(fu[1])), rad, 0, -1)
            cv2.rectangle(v, (int(fu[0]) - rad + 1, int(fu[1])), (int(fu[0]) + rad - 1, Hh), 0, -1)
            bl = cv2.GaussianBlur(r[i] * v, (0, 0), 6)
            bw = cv2.GaussianBlur(v, (0, 0), 6) + 1e-4
            hp = (r[i] - bl / bw) * v
            a = cv2.GaussianBlur(np.abs(hp), (0, 0), 1.5) / (cv2.GaussianBlur(v, (0, 0), 1.5) + 1e-4)
            v2 = cv2.erode(v, np.ones((7, 7), np.uint8)) > 0.5
            mp.append(np.where(v2, a, 0)[y0:y1, x0:x1])
            vl.append(v2[y0:y1, x0:x1])
        mp, vl = np.stack(mp), np.stack(vl)
        floor = float(np.median(mp[vl])) if vl.any() else 0.0
        mp = np.clip(mp - floor, 0, None) * vl
        out[k] = dict(K=K, M=Mx, x0=int(x0), y0=int(y0), maps=mp.astype(np.float32), valid=vl)
        nT = len(ts)
        log(f"  {k}: crop {mp.shape[2]}x{mp.shape[1]} px, {vl.mean() * 100:.0f}% usable, noise floor {floor:.4f}")
    return dict(cams=out, lo=lo, hi=hi, F=F, t=np.array(ts), nT=nT)


# ---------------------------------------------------------------- model
def pick_device(pref="auto"):
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:   # make sure grid_sample forward and backward work on this Mac
            p = torch.ones(1, 2, 4, 4, device="mps", requires_grad=True)
            g = torch.zeros(1, 3, 1, 2, device="mps")
            torch.nn.functional.grid_sample(p, g, align_corners=True).sum().backward()
            return "mps"
        except Exception:
            pass
    return "cpu"


def build_model(C=12, res=(24, 48, 96), tres=30):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class KPlanes(nn.Module):
        def __init__(s):
            super().__init__()
            s.res = res
            s.pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
            s.pl = nn.ParameterList()
            for r in res:
                for a, b in s.pairs:
                    ra = r if a < 3 else tres
                    rb = r if b < 3 else tres
                    init = torch.ones(1, C, rb, ra) if b == 3 else torch.empty(1, C, rb, ra).uniform_(0.1, 0.5)
                    s.pl.append(nn.Parameter(init))
            s.mlp = nn.Sequential(nn.Linear(C * len(res), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
            nn.init.constant_(s.mlp[-1].bias, -3.0)

        def forward(s, x):
            feats, i = [], 0
            for _ in s.res:
                f = 1.0
                for a, b in s.pairs:
                    g = torch.stack([x[:, a], x[:, b]], -1).view(1, -1, 1, 2)
                    f = f * Fn.grid_sample(s.pl[i], g, align_corners=True, padding_mode="border").view(s.pl[i].shape[1], -1).T
                    i += 1
                feats.append(f)
            return Fn.softplus(s.mlp(torch.cat(feats, -1))).squeeze(-1)

        def regulariser(s):
            reg = 0
            for i, p in enumerate(s.pl):
                a, b = s.pairs[i % 6]
                if b == 3:
                    reg = reg + ((p[..., 2:, :] - 2 * p[..., 1:-1, :] + p[..., :-2, :]) ** 2).mean()
                else:
                    reg = reg + ((p[..., 1:, :] - p[..., :-1, :]) ** 2).mean() + ((p[..., :, 1:] - p[..., :, :-1]) ** 2).mean()
            return reg

    return KPlanes()


def _rays(data, device):
    import torch
    rays = {}
    lo, hi = data["lo"], data["hi"]
    for k, c in data["cams"].items():
        K, M = c["K"], c["M"]
        R, t = M[:3, :3], M[:3, 3]
        cen = -R.T @ t
        T, Hc, Wc = c["maps"].shape
        vv, uu = np.mgrid[0:Hc, 0:Wc]
        d = np.linalg.inv(K) @ np.stack([(uu + c["x0"]).ravel(), (vv + c["y0"]).ravel(), np.ones(uu.size)])
        d = (R.T @ d).T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        inv = 1 / np.where(np.abs(d) < 1e-9, 1e-9, d)
        t1, t2 = (lo - cen) * inv, (hi - cen) * inv
        tn, tf = np.max(np.minimum(t1, t2), 1), np.min(np.maximum(t1, t2), 1)
        hit = np.where(tf > tn + 1e-3)[0]
        obs = c["maps"].reshape(T, -1)[:, hit] / 0.01
        val = c["valid"].reshape(T, -1)[:, hit]
        f32 = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
        rays[k] = dict(o=f32(np.repeat(cen[None], len(hit), 0)), d=f32(d[hit]), tn=f32(tn[hit]), tf=f32(tf[hit]),
                       obs=f32(obs), val=torch.tensor(val, device=device), idx=hit, shape=(T, Hc, Wc))
    return rays


def train(data, cfg, outdir, log=print):
    import torch
    import torch.nn.functional as Fn
    fc = cfg.get("field", {})
    steps = fc.get("steps", 1500)
    device = pick_device(fc.get("device", "auto"))
    torch.manual_seed(0)
    log(f"  training on {device} for {steps} steps")
    rays = _rays(data, device)
    nT = data["nT"]
    lo = torch.tensor(data["lo"], dtype=torch.float32, device=device)
    hi = torch.tensor(data["hi"], dtype=torch.float32, device=device)
    model = build_model(res=tuple(fc.get("resolutions", [24, 48, 96])), tres=fc.get("time_resolution", 30)).to(device)
    gain = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 2e-2}, {"params": [gain], "lr": 1e-2}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-3)
    NS, B = fc.get("samples_per_ray", 40), fc.get("rays_per_step", 4096)

    def render(k, ri, ti, ns=NS, jitter=True):
        r = rays[k]
        o, d, tn, tf = r["o"][ri], r["d"][ri], r["tn"][ri], r["tf"][ri]
        u = (torch.arange(ns, device=device) + (torch.rand(len(ri), ns, device=device) if jitter else 0.5)) / ns
        tt = tn[:, None] + (tf - tn)[:, None] * u
        pts = o[:, None] + d[:, None] * tt[..., None]
        xn = (pts - lo) / (hi - lo) * 2 - 1
        tnorm = (ti.float() / max(nT - 1, 1) * 2 - 1)[:, None, None].expand(-1, ns, 1)
        sig = model(torch.cat([xn, tnorm], -1).view(-1, 4)).view(len(ri), ns)
        return (sig * ((tf - tn) / ns)[:, None]).sum(1) * 20 * torch.exp(gain), sig

    def predict(k, ti):
        r = rays[k]
        T, Hc, Wc = r["shape"]
        P = np.zeros(Hc * Wc, np.float32)
        out = []
        with torch.no_grad():
            for ch in torch.split(torch.arange(len(r["idx"]), device=device), 8192):
                pr, _ = render(k, ch, torch.full((len(ch),), ti, device=device), ns=64, jitter=False)
                out.append(pr)
        P[r["idx"]] = torch.cat(out).cpu().numpy() * 0.01
        return P.reshape(Hc, Wc)

    cams = list(rays)
    snaps = []
    mid = nT // 2
    t0 = time.time()
    for step in range(steps):
        loss = 0
        for k in cams:
            r = rays[k]
            ri = torch.randint(0, len(r["idx"]), (B // len(cams),), device=device)
            ti = torch.randint(0, nT, (B // len(cams),), device=device)
            obs, m = r["obs"][ti, ri], r["val"][ti, ri]
            pred, sig = render(k, ri, ti)
            w = m.float() * (1 + 2 * (obs > 0.6).float())
            loss = loss + (w * Fn.smooth_l1_loss(pred, obs, reduction="none", beta=0.5)).sum() / (w.sum() + 1e-6) + 2e-3 * sig.mean()
        total = loss + 1e-2 * model.regulariser()
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            log(f"    step {step:5d}  fit {loss.item():.4f}  {time.time() - t0:.0f}s")
        if step % max(1, steps // 30) == 0 or step == steps - 1:
            snaps.append((step, {k: predict(k, mid) for k in cams}))
    torch.save(dict(state=model.state_dict(), gain=gain.detach().cpu()), os.path.join(outdir, "field.pt"))

    # export: density volume per frame, and every camera's predicted view per frame
    nx, ny, nz = fc.get("export_grid", [48, 84, 48])
    with torch.no_grad():
        g = [torch.linspace(-1, 1, n, device=device) for n in (nx, ny, nz)]
        G = torch.stack(torch.meshgrid(*g, indexing="ij"), -1).view(-1, 3)
        vols = []
        for ti in range(nT):
            tt = torch.full((G.shape[0], 1), ti / max(nT - 1, 1) * 2 - 1, device=device)
            vols.append(torch.cat([model(torch.cat([ch, tt[:len(ch)]], -1)) for ch in torch.split(G, 65536)]).view(nx, ny, nz).cpu().numpy())
    vol = np.stack(vols).astype(np.float16)
    preds = {k: np.stack([predict(k, ti) for ti in range(nT)]) for k in cams}
    return vol, preds, snaps


def carve(vol, data, thr=0.002, dilate_px=7):
    """Soft space carving: scale density by how clearly every camera saw heat along the ray through it
    (1 = clearly seen by all, fading to 0 if any camera saw nothing there). Nothing is kept below the flame."""
    T, nx, ny, nz = vol.shape
    lo, hi, F = data["lo"], data["hi"], data["F"]
    X = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], n) for i, n in enumerate((nx, ny, nz))], indexing="ij"), -1).reshape(-1, 3)
    w = np.ones((T, X.shape[0]), np.float32)
    k = np.ones((dilate_px, dilate_px), np.uint8)
    for c in data["cams"].values():
        p = c["K"] @ (c["M"][:3, :3] @ X.T + c["M"][:3, 3:])
        u, v = p[0] / p[2] - c["x0"], p[1] / p[2] - c["y0"]
        _, Hc, Wc = c["maps"].shape
        inside = (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
        ui, vi = np.clip(np.round(u).astype(int), 0, Wc - 1), np.clip(np.round(v).astype(int), 0, Hc - 1)
        for t in range(T):
            m = cv2.GaussianBlur(cv2.dilate(c["maps"][t], k), (0, 0), 2.0)
            w[t] *= inside * np.clip(m[vi, ui] / thr, 0, 1)
    w *= (X[:, 1] < F[1] - 0.005)[None]
    return (vol.astype(np.float32) * w.reshape(T, nx, ny, nz)).astype(np.float16)
