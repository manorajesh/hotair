"""4D neural field of the air, fitted with a physical shadowgraph forward model.

Each camera sees the retroreflective sheet through the plume. To first order a shadowgram is the 2D Laplacian of the
density integrated along each line of sight:   I / I_ref - 1  ~  kappa * Laplacian( integral of rho dl ).
So instead of fitting a blurred "heat energy" map we fit a non-negative density field rho(x, y, z, t) whose rendered
Laplacian matches every camera's *signed* shadowgram at full resolution. Training renders small patches of adjacent
rays so the Laplacian can be taken on the rendered projection.

Model: multiscale K-Planes (six 2D feature planes per scale: xy xz yz xt yt zt) with per-axis resolution set in
millimetres, features multiplied within a scale, concatenated across scales, small MLP, softplus.
A per-frame visual hull (where each camera saw any signal, smoothed over a few frames) multiplies the density during
training and rendering, which removes most limited-angle "walls" without per-frame flicker.
"""
import os
import time

import cv2
import numpy as np

from .maps import masked_highpass
from .media import decode, probe

BINOM = np.array([0.25, 0.5, 0.25], np.float32)


def _blur3(img):
    return cv2.sepFilter2D(img, -1, BINOM, BINOM, borderType=cv2.BORDER_REPLICATE)


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


def prepare(cfg, calib, offsets, camdirs, log=print):
    """Signed, noise-normalised shadowgrams of the field window, cropped to each camera's view of the 3D box."""
    fc = cfg.get("field", {})
    pr = cfg.get("processing", {})
    fps = pr.get("field_fps", 30)
    fw = pr.get("field_width")                     # None: native resolution
    tracks = {}
    for k, d in camdirs.items():
        z = np.load(os.path.join(d, "maps.npz"))
        tracks[k] = {round(float(t), 3): np.array(uv) for t, uv in zip(z["track_t"], z["track_uv"])}
    f0, f1 = cfg["take"]["field_window"]
    r0, r1 = cfg["take"]["reference"]
    F = np.array(fc["box_center_m"]) if fc.get("box_center_m") else flame_center(calib, tracks, f0, f1)
    w, h, d = fc.get("box_size_m", [0.26, 0.45, 0.26])
    below = fc.get("box_below_flame_m", 0.03)
    lo = F + np.array([-w / 2, -(h - below), -d / 2])        # y points down: the plume rises towards -y
    hi = F + np.array([w / 2, below, d / 2])
    log(f"  flame at {np.round(F, 3)} m; 3D box {w * 100:.0f} x {h * 100:.0f} x {d * 100:.0f} cm; {fps} fps")
    corners = np.array([[x, y, zz] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for zz in (lo[2], hi[2])])
    out = {}
    ts = None
    for k, cam in cfg["cameras"].items():
        c = calib["cameras"].get(k, {})
        if "R" not in c:
            log(f"  {k}: not calibrated, skipped")
            continue
        path, off = cam["take"], offsets[k]
        info = probe(path)
        W = int(fw or info["width"])
        s = W / info["width"]
        px = W / 1920                               # pixel sizes in the config are for 1920-wide frames
        K = np.array(c["K"], float).copy()
        K[:2, :2] *= s
        K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
        R, t = np.array(c["R"], float), np.array(c["t"], float)
        ref = np.median(np.stack([g.astype(np.float32) for g in decode(path, r0 + off, r1 - r0, 15, fw)]), 0)
        Hh, Ww = ref.shape
        pc = K @ (R @ corners.T + t[:, None])
        pc = (pc[:2] / pc[2]).T
        x0, y0 = np.maximum(np.floor(pc.min(0)).astype(int) - 4, 0)
        x1, y1 = np.minimum(np.ceil(pc.max(0)).astype(int) + 5, [Ww, Hh])
        edge = max(3, int(round(28 * px)))
        sheet = (ref > 0.45 * np.percentile(ref, 99)).astype(np.uint8)
        sheet_core = (cv2.erode(sheet, np.ones((edge, edge), np.uint8)) > 0)[y0:y1, x0:x1]
        refc = ref[y0:y1, x0:x1]
        pf = K @ (R @ F + t)
        pf = pf[:2] / pf[2]
        tk = tracks[k]
        tt = np.array(sorted(tk)) if tk else np.array([])
        rad = int(round(fc.get("flame_mask_px", 28) * px))
        hp_sigma = fc.get("highpass_px", 30) * px
        sgs, vals, tlist = [], [], []
        for i, g in enumerate(decode(path, f0 + off, f1 - f0, fps, fw)):
            ti = f0 + i / fps
            gc = g[y0:y1, x0:x1].astype(np.float32)
            r = (gc + 1) / (refc + 1) - 1
            v = (sheet_core & (gc > 0.45 * refc)).astype(np.uint8)
            v = cv2.erode(v, np.ones((edge, edge), np.uint8))
            fu = pf
            if len(tt):
                j = int(np.argmin(np.abs(tt - ti)))
                if abs(tt[j] - ti) < 0.2:
                    fu = tk[tt[j]] * s
            fx, fy = int(fu[0]) - x0, int(fu[1]) - y0
            cv2.circle(v, (fx, fy), rad, 0, -1)
            cv2.rectangle(v, (fx - rad + 1, fy), (fx + rad - 1, v.shape[0]), 0, -1)
            sg = masked_highpass(r, v > 0, hp_sigma)
            sg -= np.median(sg[v > 0]) if v.any() else 0
            v2 = cv2.erode(v, np.ones((5, 5), np.uint8)) > 0
            sgs.append(np.where(v2, _blur3(sg), 0).astype(np.float32))
            vals.append(v2)
            tlist.append(ti)
        sg, val = np.stack(sgs), np.stack(vals)
        noise = 1.4826 * float(np.median(np.abs(sg[val]))) + 1e-9
        sg = np.clip(sg / noise, -8, 8)
        # visual hull: where this camera saw signal (smoothed in space, max over a few frames), dilated
        hs = fc.get("hull_smooth_px", 8) * px
        dil = max(3, int(round(fc.get("hull_dilate_px", 24) * px)) | 1)
        E = np.stack([cv2.GaussianBlur(np.abs(a), (0, 0), hs) / (cv2.GaussianBlur(m.astype(np.float32), (0, 0), hs) + 1e-3)
                      for a, m in zip(sg, val)])
        e = E[val]
        med = float(np.median(e))
        thr = med + fc.get("hull_threshold", 6.0) * 1.4826 * float(np.median(np.abs(e - med)))   # robust z-score
        min_area = fc.get("hull_min_area_px", 150) * px * px
        sig = (E > thr) & val
        for i in range(len(sig)):                     # drop isolated noise specks
            n, lab, st, _ = cv2.connectedComponentsWithStats(sig[i].astype(np.uint8))
            sig[i] = np.isin(lab, 1 + np.where(st[1:, cv2.CC_STAT_AREA] >= min_area)[0])
        del E
        tw = fc.get("hull_frames", 2)
        sig_t = np.stack([sig[max(0, i - tw):i + tw + 1].any(0) for i in range(len(sig))])
        hull = []
        known = np.stack([cv2.dilate(m.astype(np.uint8), np.ones((dil, dil), np.uint8)) for m in val]) > 0
        for i in range(len(sig_t)):
            hm = cv2.dilate(sig_t[i].astype(np.uint8), np.ones((dil, dil), np.uint8)).astype(np.float32)
            hm = np.where(known[i] | sig_t[i], hm, 1.0)          # nothing known there (flame, hand, off-sheet): don't carve
            hull.append(cv2.GaussianBlur(hm, (0, 0), 2 * hs))
        hull = np.clip(np.stack(hull), 0, 1)
        sigpix = np.argwhere(sig).astype(np.int32)
        out[k] = dict(K=K, R=R, t=t, x0=int(x0), y0=int(y0), sg=sg.astype(np.float16), valid=val,
                      hull=(hull * 255).astype(np.uint8), sigpix=sigpix, noise=noise)
        ts = np.array(tlist)
        log(f"  {k}: crop {sg.shape[2]}x{sg.shape[1]} px x {len(sg)} frames, {val.mean() * 100:.0f}% usable, "
            f"noise {noise:.4f}, signal in {sig.sum() / max(val.sum(), 1) * 100:.1f}% of usable pixels")
    nT = min(len(c["sg"]) for c in out.values())
    for c in out.values():
        for key in ("sg", "valid", "hull"):
            c[key] = c[key][:nT]
        c["sigpix"] = c["sigpix"][c["sigpix"][:, 0] < nT]
    return dict(cams=out, lo=lo, hi=hi, F=F, t=ts[:nT], nT=nT, fps=fps)


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


def model_spec(data, fc):
    size = data["hi"] - data["lo"]
    return dict(size=[float(x) for x in size], nT=int(data["nT"]), C=fc.get("channels", 8),
                vox_mm=list(fc.get("vox_mm", [10, 5, 2.5, 1.25])), t_frac=list(fc.get("time_fraction", [0.25, 0.5, 1, 1])))


def build_model(spec):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    C = spec["C"]
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]

    class KPlanes(nn.Module):
        def __init__(s):
            super().__init__()
            s.pl = nn.ParameterList()
            s.time_plane = []
            s.nscale = len(spec["vox_mm"])
            for vox, tf in zip(spec["vox_mm"], spec["t_frac"]):
                res = [max(4, int(round(L * 1000 / vox)) + 1) for L in spec["size"]] + [max(4, int(round(spec["nT"] * tf)))]
                for a, b in pairs:
                    if b == 3:
                        init = torch.ones(1, C, res[b], res[a])
                    else:
                        init = torch.empty(1, C, res[b], res[a]).uniform_(0.1, 0.5)
                    s.pl.append(nn.Parameter(init))
                    s.time_plane.append(b == 3)
            s.mlp = nn.Sequential(nn.Linear(C * s.nscale, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
            nn.init.constant_(s.mlp[-1].bias, -3.0)

        def forward(s, x):
            feats, i = [], 0
            for _ in range(s.nscale):
                f = 1.0
                for a, b in pairs:
                    g = torch.stack([x[:, a], x[:, b]], -1).view(1, -1, 1, 2)
                    f = f * Fn.grid_sample(s.pl[i], g, align_corners=True, padding_mode="border").view(C, -1).T
                    i += 1
                feats.append(f)
            return Fn.softplus(s.mlp(torch.cat(feats, -1))).squeeze(-1)

        def regulariser(s):
            sp, tm = 0, 0
            for p, is_t in zip(s.pl, s.time_plane):
                if is_t:   # time runs along dim -2: smooth motion (1st and 2nd differences)
                    d1 = p[..., 1:, :] - p[..., :-1, :]
                    tm = tm + (d1 ** 2).mean() + ((d1[..., 1:, :] - d1[..., :-1, :]) ** 2).mean()
                else:
                    sp = sp + ((p[..., 1:, :] - p[..., :-1, :]) ** 2).mean() + ((p[..., :, 1:] - p[..., :, :-1]) ** 2).mean()
            return sp, tm

    return KPlanes()


class Scene:
    """Everything on the device needed to render rays through the field: box, hull, cameras."""

    def __init__(s, data, device, use_hull=True):
        import torch
        s.torch = torch
        s.dev = device
        f32 = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
        s.lo, s.hi = f32(data["lo"]), f32(data["hi"])
        s.Fy = float(data["F"][1])
        s.nT = data["nT"]
        s.cams = {}
        for k, c in data["cams"].items():
            T, Hc, Wc = c["sg"].shape
            s.cams[k] = dict(KR=f32(c["K"] @ c["R"]), Kt=f32(c["K"] @ c["t"]), Kinv=f32(np.linalg.inv(c["K"])), Rm=f32(c["R"]),
                             cen=f32(-c["R"].T @ c["t"]), x0=c["x0"], y0=c["y0"], H=Hc, W=Wc,
                             hull=f32(c["hull"].astype(np.float32) / 255)[None, None] if use_hull else None)
        s.use_hull = use_hull

    def tnorm(s, ti):
        return ti.float() / max(s.nT - 1, 1) * 2 - 1

    def weight(s, pts, tn):
        """Hull (product over cameras) and 'nothing below the flame', for world points (N,3) at normalised times (N,)."""
        torch = s.torch
        w = torch.sigmoid((s.Fy - 0.005 - pts[:, 1]) / 0.002)
        if not s.use_hull:
            return w
        for c in s.cams.values():
            p = pts @ c["KR"].T + c["Kt"]
            u = (p[:, 0] / p[:, 2] - c["x0"]) / (c["W"] - 1) * 2 - 1
            v = (p[:, 1] / p[:, 2] - c["y0"]) / (c["H"] - 1) * 2 - 1
            g = torch.stack([u, v, tn], -1).view(1, -1, 1, 1, 3)
            w = w * torch.nn.functional.grid_sample(c["hull"], g, align_corners=True, padding_mode="border").view(-1)
        return w

    def slab(s, o, d):
        torch = s.torch
        inv = 1 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t1, t2 = (s.lo - o) * inv, (s.hi - o) * inv
        tn = torch.minimum(t1, t2).amax(-1).clamp(min=0)
        tf = torch.maximum(t1, t2).amin(-1)
        return tn, torch.maximum(tf, tn)

    def cam_rays(s, k, u, v):
        """Rays for crop pixel coordinates (u, v) of camera k."""
        c = s.cams[k]
        torch = s.torch
        pix = torch.stack([u.float() + c["x0"], v.float() + c["y0"], torch.ones_like(u, dtype=torch.float32)], -1)
        d = (pix @ c["Kinv"].T) @ c["Rm"]
        d = d / d.norm(dim=-1, keepdim=True)
        o = c["cen"].expand_as(d)
        return o, d

    def march(s, model, o, d, tn_frame, ns, jitter):
        """Samples along rays. Returns density (N, ns) and step length (N,). `jitter` (N,) in [0,1) is shared per patch."""
        torch = s.torch
        tn, tf = s.slab(o, d)
        u = (torch.arange(ns, device=s.dev) + jitter[:, None]) / ns
        tt = tn[:, None] + (tf - tn)[:, None] * u
        pts = (o[:, None] + d[:, None] * tt[..., None]).reshape(-1, 3)
        tq = tn_frame[:, None].expand(-1, ns).reshape(-1)
        w = s.weight(pts, tq)
        keep = torch.nonzero(w > 1e-3).squeeze(1)          # only evaluate the field inside the hull
        xn = (pts[keep] - s.lo) / (s.hi - s.lo) * 2 - 1
        sig = torch.zeros_like(w)
        if len(keep):
            sig = sig.index_put((keep,), model(torch.cat([xn, tq[keep, None]], -1)) * w[keep])
        return sig.view(-1, ns), (tf - tn) / ns


def laplacian_of(P):
    """3x3 binomial blur then 5-point Laplacian, both 'valid': (n, H, W) -> (n, H-4, W-4)."""
    import torch
    import torch.nn.functional as Fn
    b = torch.tensor(BINOM, device=P.device)
    k1 = (b[:, None] * b[None]).view(1, 1, 3, 3)
    k2 = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]], device=P.device).view(1, 1, 3, 3)
    return Fn.conv2d(Fn.conv2d(P[:, None], k1), k2)[:, 0]


def predict_camera(model, scene, gain, k, ti, ns=48, chunk=32768):
    """Full-crop synthetic shadowgram (noise units) for camera k at frame ti."""
    torch = scene.torch
    c = scene.cams[k]
    H, W = c["H"], c["W"]
    vv, uu = torch.meshgrid(torch.arange(-2, H + 2, device=scene.dev), torch.arange(-2, W + 2, device=scene.dev), indexing="ij")
    u, v = uu.reshape(-1), vv.reshape(-1)
    P = []
    with torch.no_grad():
        for i in range(0, len(u), chunk):
            o, d = scene.cam_rays(k, u[i:i + chunk], v[i:i + chunk])
            n = len(o)
            sig, dt = scene.march(model, o, d, scene.tnorm(torch.full((n,), ti, device=scene.dev)), ns,
                                  torch.full((n,), 0.5, device=scene.dev))
            P.append((sig * dt[:, None]).sum(1))
        P = torch.cat(P).view(1, H + 4, W + 4)
        return (laplacian_of(P)[0] * gain(k)).cpu().numpy()


def train(data, cfg, outdir, log=print):
    import torch
    import torch.nn.functional as Fn
    fc = cfg.get("field", {})
    steps = fc.get("steps", 2000)
    device = pick_device(fc.get("device", "auto"))
    torch.manual_seed(0)
    np.random.seed(0)
    spec = model_spec(data, fc)
    model = build_model(spec).to(device)
    scene = Scene(data, device)
    cams = list(data["cams"])
    nT = data["nT"]
    S = fc.get("patch_px", 32)
    Rn = S + 4
    npatch = fc.get("patches_per_camera", 12)
    NS = fc.get("samples_per_ray", 48)
    sg = {k: torch.tensor(data["cams"][k]["sg"], device=device) for k in cams}
    val = {k: torch.tensor(data["cams"][k]["valid"], device=device) for k in cams}
    sigpix = {k: torch.tensor(data["cams"][k]["sigpix"], dtype=torch.long, device=device) for k in cams}
    log(f"  training on {device} for {steps} steps: {npatch} patches of {S}x{S} px per camera per step, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    g0 = torch.tensor(float(np.log(fc.get("gain_init", 5000.0))), device=device, requires_grad=True)
    gk = torch.zeros(len(cams), device=device, requires_grad=True)
    gain = lambda k: torch.exp(g0 + gk[cams.index(k)])
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": fc.get("lr", 2e-2)}, {"params": [g0, gk], "lr": 1e-2}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=fc.get("lr", 2e-2) / 20)
    lam_s, lam_t, lam_sp = fc.get("smooth_space", 1e-3), fc.get("smooth_time", 1e-2), fc.get("sparsity", 1e-3)
    frac_sig = fc.get("signal_patch_fraction", 0.75)
    ar = torch.arange(S, device=device)
    snaps = []
    snap_every = max(1, steps // 20)
    mid = nT // 2
    t0 = time.time()
    hist = []
    for step in range(steps):
        loss = 0
        sp_term = 0
        for k in cams:
            c = scene.cams[k]
            H, W = c["H"], c["W"]
            nsig = int(round(npatch * frac_sig)) if len(sigpix[k]) else 0
            pick = sigpix[k][torch.randint(0, max(len(sigpix[k]), 1), (nsig,), device=device)] if nsig else \
                torch.zeros((0, 3), dtype=torch.long, device=device)
            rnd = torch.stack([torch.randint(0, nT, (npatch - nsig,), device=device),
                               torch.randint(0, H, (npatch - nsig,), device=device),
                               torch.randint(0, W, (npatch - nsig,), device=device)], -1)
            tyx = torch.cat([pick, rnd])
            ti = tyx[:, 0]
            py = (tyx[:, 1] - Rn // 2).clamp(0, H - Rn)
            px = (tyx[:, 2] - Rn // 2).clamp(0, W - Rn)
            ry = torch.arange(Rn, device=device)
            vv = (py[:, None, None] + ry[None, :, None]).expand(-1, Rn, Rn)
            uu = (px[:, None, None] + ry[None, None, :]).expand(-1, Rn, Rn)
            o, d = scene.cam_rays(k, uu.reshape(-1), vv.reshape(-1))
            tq = scene.tnorm(ti[:, None].expand(-1, Rn * Rn).reshape(-1))
            jit = torch.rand(npatch, device=device)[:, None].expand(-1, Rn * Rn).reshape(-1)
            sig, dt = scene.march(model, o, d, tq, NS, jit)
            P = (sig * dt[:, None]).sum(1).view(npatch, Rn, Rn)
            pred = laplacian_of(P) * gain(k)
            ys = (py + 2)[:, None, None] + ar[None, :, None]
            xs = (px + 2)[:, None, None] + ar[None, None, :]
            tgt = sg[k][ti[:, None, None], ys, xs].float()
            w = val[k][ti[:, None, None], ys, xs].float()
            loss = loss + (w * Fn.smooth_l1_loss(pred, tgt, reduction="none", beta=1.0)).sum() / (w.sum() + 1e-6)
            sp_term = sp_term + sig.mean()
        rs, rt = model.regulariser()
        total = loss + lam_sp * sp_term + lam_s * rs + lam_t * rt + 1e-2 * (gk ** 2).sum()
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()
        hist.append(loss.item() / len(cams))
        if step % 100 == 0 or step == steps - 1:
            log(f"    step {step:5d}  fit {np.mean(hist[-100:]):.4f}  {time.time() - t0:.0f}s")
        if step % snap_every == 0 or step == steps - 1:
            model.eval()
            snaps.append((step, {k: predict_camera(model, scene, gain, k, mid, ns=32) for k in cams}))
            model.train()
    gains = {k: float(gain(k).item()) for k in cams}
    torch.save(dict(state=model.state_dict(), spec=spec, gains=gains), os.path.join(outdir, "field.pt"))
    log("  gains " + ", ".join(f"{k} {v:.0f}" for k, v in gains.items()))
    return model, scene, snaps


def load(path, data, device=None):
    import torch
    device = device or pick_device()
    ck = torch.load(path, map_location=device)
    model = build_model(ck["spec"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, Scene(data, device), (lambda k: ck["gains"][k])


def export_volume(model, scene, data, vox_mm=4.0, chunk=262144):
    """Density on a regular grid for every frame: (time, x, y, z), float16, hull applied."""
    torch = scene.torch
    size = data["hi"] - data["lo"]
    n = [max(2, int(round(L * 1000 / vox_mm)) + 1) for L in size]
    g = [torch.linspace(float(data["lo"][i]), float(data["hi"][i]), n[i], device=scene.dev) for i in range(3)]
    G = torch.stack(torch.meshgrid(*g, indexing="ij"), -1).view(-1, 3)
    xn = (G - scene.lo) / (scene.hi - scene.lo) * 2 - 1
    vols = []
    with torch.no_grad():
        for ti in range(data["nT"]):
            out = []
            for i in range(0, len(G), chunk):
                tq = scene.tnorm(torch.full((len(G[i:i + chunk]),), ti, device=scene.dev))
                out.append(model(torch.cat([xn[i:i + chunk], tq[:, None]], -1)) * scene.weight(G[i:i + chunk], tq))
            vols.append(torch.cat(out).view(*n).cpu().numpy().astype(np.float16))
    return np.stack(vols)
