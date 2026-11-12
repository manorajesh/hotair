"""Camera calibration.

1. Find the ChArUco board in every calibration clip, sampled at the same moments on the reference clock.
2. Focal length per camera (phone lenses: centred, no distortion).
3. Relative position of every camera that shares board views with another (stereo, still frames only).
4. Cameras that never read the board are placed later from the flame in the take (see place_from_points).

World coordinates = the anchor camera's coordinates (metres, x right, y down, z forward).
"""
import json
from collections import deque

import cv2
import numpy as np

from .media import decode, probe


def make_board(b):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b.get("dictionary", "DICT_5X5_100")))
    sq = b["square_mm"] / 1000.0
    return cv2.aruco.CharucoBoard((b["squares_x"], b["squares_y"]), sq, sq * b.get("marker_ratio", 0.75), d)


def _detector(board):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.01
    return cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), p)


def detect_clip(path, offset, t0, t1, fps, board, enhance=False, min_corners=8, log=print):
    """Board corners at reference-clock times t0..t1. Returns list of (t_ref, corners Nx2, ids N)."""
    det = _detector(board)
    fstart = max(0.0, t0 + offset)
    out = []
    clahe = cv2.createCLAHE(3.0, (8, 8))
    for i, g in enumerate(decode(path, fstart, t1 - t0, fps)):
        t_ref = fstart - offset + i / fps
        img, s = g, 1.0
        if enhance:
            img, s = clahe.apply(cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)), 2.0
        cc, ci, _, _ = det.detectBoard(img)
        if ci is not None and len(ci) >= min_corners:
            out.append((t_ref, cc.reshape(-1, 2) / s, ci.ravel()))
    return out


def focal_only(dets, board, size):
    allc = board.getChessboardCorners().astype(np.float32)
    obj = [allc[ids] for _, _, ids in dets]
    img = [c.astype(np.float32) for _, c, _ in dets]
    w, h = size
    K0 = np.array([[1.4 * w, 0, w / 2], [0, 1.4 * w, h / 2], [0, 0, 1]], float)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K, _, rv, tv = cv2.calibrateCamera(obj, img, size, K0, np.zeros(5), flags=flags)
    errs = []
    for o, i, r, t in zip(obj, img, rv, tv):
        p, _ = cv2.projectPoints(o, r, t, K, np.zeros(5))
        errs.append(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, 1))))
    keep = np.array(errs) < max(1.5, np.percentile(errs, 85))
    rms, K, _, _, tv = cv2.calibrateCamera([o for o, k in zip(obj, keep) if k], [i for i, k in zip(img, keep) if k],
                                           size, K, np.zeros(5), flags=flags)
    return K, float(rms), float(np.median([np.linalg.norm(t) for t in tv]))


def _motion(dets, k, dt):
    t, c, ids = dets[k]
    m = dict(zip(ids, c))
    v = []
    for j in (k - 1, k + 1):
        if 0 <= j < len(dets) and abs(dets[j][0] - t) < 1.5 * dt:
            mm = dict(zip(dets[j][2], dets[j][1]))
            com = [i for i in ids if i in mm]
            if len(com) > 5:
                v.append(np.median([np.linalg.norm(m[i] - mm[i]) for i in com]))
    return max(v) if v else 1e9


def stereo(dA, dB, KA, KB, board, size, dt, max_motion=4.0, log=print):
    """Pose of camera B relative to camera A from simultaneous, nearly still board views."""
    allc = board.getChessboardCorners().astype(np.float32)
    tB = np.array([x[0] for x in dB])
    items = []
    for k, (ta, ca, ia) in enumerate(dA):
        if not len(tB):
            break
        j = int(np.argmin(np.abs(tB - ta)))
        if abs(tB[j] - ta) > 0.5 * dt or _motion(dA, k, dt) > max_motion or _motion(dB, j, dt) > max_motion:
            continue
        com = np.intersect1d(ia, dB[j][2])
        if len(com) < 10:
            continue
        mA, mB = dict(zip(ia, ca)), dict(zip(dB[j][2], dB[j][1]))
        items.append((ta, allc[com], np.array([mA[i] for i in com], np.float32), np.array([mB[i] for i in com], np.float32)))
    sel = []
    for it in items:                                   # roughly one frame per pose
        if not sel or it[0] - sel[-1][0] > 0.2:
            sel.append(it)
    if len(sel) < 4:
        return None
    z = np.zeros(5)
    while True:
        rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate([x[1] for x in sel], [x[2] for x in sel], [x[3] for x in sel],
                                                         KA, z, KB, z, size, flags=cv2.CALIB_FIX_INTRINSIC)
        errs = []
        for _, o, pa, pb in sel:
            _, rv, tv = cv2.solvePnP(o, pa, KA, z, flags=cv2.SOLVEPNP_IPPE)
            Rb = cv2.Rodrigues(rv)[0]
            P, _ = cv2.projectPoints(o, cv2.Rodrigues(R @ Rb)[0], R @ tv + T, KB, z)
            errs.append(np.sqrt(np.mean(np.sum((P.reshape(-1, 2) - pb) ** 2, 1))))
        w = int(np.argmax(errs))
        if (errs[w] < 3 and rms < 1.2) or len(sel) <= 5:
            break
        sel.pop(w)
    return dict(R=R, T=T.ravel(), rms=float(rms), views=len(sel))


def calibrate(cfg, offsets, log=print, cache=None):
    """Board-based calibration. offsets: {cam: seconds} for the calibration clips (reference clock)."""
    cams = cfg["cameras"]
    board = make_board(cfg["board"])
    fps = cfg["calibration"].get("detect_fps", 10)
    info = {k: probe(c["calib"]) for k, c in cams.items()}
    size = (info[next(iter(cams))]["width"], info[next(iter(cams))]["height"])
    t0 = max(-offsets[k] for k in cams)
    t1 = min(info[k]["duration"] - offsets[k] for k in cams)
    log(f"  shared calibration window on the reference clock: {t0:.1f}-{t1:.1f} s")
    import os, pickle
    key = dict(fps=fps, t0=round(t0, 3), t1=round(t1, 3), board=cfg["board"], files={k: c["calib"] for k, c in cams.items()})
    cached = pickle.load(open(cache, "rb")) if cache and os.path.exists(cache) else None
    if cached and cached["key"] == key:
        log("  using cached board detections")
        dets = cached["dets"]
    else:
        dets = {}
    for k, c in cams.items():
        if k in dets:
            log(f"  {k}: board found in {len(dets[k])} frames")
            continue
        d = detect_clip(c["calib"], offsets[k], t0, t1, fps, board)
        if len(d) < 10:
            log(f"  {k}: board found in only {len(d)} frames, retrying with enlarged and contrast-boosted frames")
            d2 = detect_clip(c["calib"], offsets[k], t0, t1, fps, board, enhance=True)
            d = d2 if len(d2) > len(d) else d
        dets[k] = d
        log(f"  {k}: board found in {len(d)} frames")
    if cache:
        pickle.dump(dict(key=key, dets=dets), open(cache, "wb"))
    K = {}
    for k, d in dets.items():
        if len(d) >= 8:
            K[k], rms, dist = focal_only(d, board, size)
            log(f"  {k}: focal length {K[k][0, 0]:.0f} px  (fit {rms:.2f} px, board ~{dist:.1f} m away)")
    # pairwise relative poses, then chain them from the anchor camera
    rel = {}
    ks = list(K)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            r = stereo(dets[a], dets[b], K[a], K[b], board, size, 1.0 / fps, log=log)
            if r:
                rel[(a, b)] = r
                log(f"  {a}<->{b}: {r['views']} shared still views, fit {r['rms']:.2f} px, "
                    f"{np.linalg.norm(r['T']):.2f} m apart")
    anchor = cfg["calibration"].get("anchor") or max(ks, key=lambda k: len(dets[k])) if ks else None
    pose = {}
    if anchor:
        pose[anchor] = (np.eye(3), np.zeros(3))
        q = deque([anchor])
        while q:
            a = q.popleft()
            for (x, y), r in rel.items():
                for src, dst, R, T in ((x, y, r["R"], r["T"]), (y, x, r["R"].T, -r["R"].T @ r["T"])):
                    if src == a and dst not in pose:
                        Ra, ta = pose[a]
                        pose[dst] = (R @ Ra, R @ ta + T)
                        q.append(dst)
    result = dict(anchor=anchor, image_size=list(size), cameras={})
    for k in cams:
        e = dict(method="board" if k in pose else "uncalibrated")
        if k in K:
            e["K"] = K[k].tolist()
        if k in pose:
            e["R"], e["t"] = pose[k][0].tolist(), pose[k][1].tolist()
        result["cameras"][k] = e
    missing = [k for k in cams if k not in pose]
    if missing:
        log(f"  not placed by the board: {', '.join(missing)} -> will be placed from the flame in the take")
    return result, dets


def triangulate(obs, Ps):
    """Multi-view DLT. obs: list of (u,v); Ps: list of 3x4."""
    A = []
    for (u, v), P in zip(obs, Ps):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def place_from_points(calib, tracks, log=print, max_reproj=6.0, same_lens=True):
    """Place uncalibrated cameras using a moving bright point (the flame) seen by everyone.

    tracks: {cam: {t_ref(rounded): (u, v) full-res}}.
    """
    cams = calib["cameras"]
    placed = [k for k, c in cams.items() if "R" in c]
    todo = [k for k, c in cams.items() if "R" not in c]
    if not todo:
        return calib
    if len(placed) < 2:
        raise RuntimeError("Need at least two board-calibrated cameras to place the others from the flame.")
    fs = [cams[k]["K"][0][0] for k in placed]
    w, h = calib["image_size"]
    P = {k: np.array(cams[k]["K"]) @ np.c_[np.array(cams[k]["R"]), np.array(cams[k]["t"])] for k in placed}
    times = sorted(set.intersection(*[set(tracks[k]) for k in placed]))
    X, T = [], []
    for t in times:
        x = triangulate([tracks[k][t] for k in placed], [P[k] for k in placed])
        err = max(np.linalg.norm((P[k] @ np.r_[x, 1])[:2] / (P[k] @ np.r_[x, 1])[2] - tracks[k][t]) for k in placed)
        if err < max_reproj:
            X.append(x)
            T.append(t)
    X = np.array(X, np.float32)
    log(f"  flame located in 3D at {len(X)} moments (spread {np.round(X.std(0) * 100, 1)} cm)")
    for k in todo:
        own = cams[k].get("K")
        if same_lens or own is None:
            if own is not None and abs(own[0][0] / np.median(fs) - 1) > 0.05:
                log(f"  {k}: ignoring its own focal length ({own[0][0]:.0f} px, from too few board views); "
                    f"using {np.median(fs):.0f} px from the other phones")
            K = np.array([[np.median(fs), 0, w / 2], [0, np.median(fs), h / 2], [0, 0, 1]], float)
        else:
            K = np.array(own, float)
        ts = [t for t in T if t in tracks[k]]
        Xi = np.array([X[T.index(t)] for t in ts], np.float32)
        U = np.array([tracks[k][t] for t in ts], np.float32)
        if len(Xi) < 12:
            log(f"  {k}: not enough shared flame sightings ({len(Xi)}) - left uncalibrated")
            continue
        ok, rv, tv, inl = cv2.solvePnPRansac(Xi, U, K, np.zeros(5), reprojectionError=8, iterationsCount=5000)
        if not ok or inl is None or len(inl) < 8:
            log(f"  {k}: could not place from the flame - left uncalibrated")
            continue
        rv, tv = cv2.solvePnPRefineLM(Xi[inl.ravel()], U[inl.ravel()], K, np.zeros(5), rv, tv)
        Pp, _ = cv2.projectPoints(Xi[inl.ravel()], rv, tv, K, np.zeros(5))
        e = float(np.median(np.linalg.norm(Pp.reshape(-1, 2) - U[inl.ravel()], axis=1)))
        cams[k].update(K=K.tolist(), R=cv2.Rodrigues(rv)[0].tolist(), t=tv.ravel().tolist(), method="flame",
                       flame_fit_px=e)
        log(f"  {k}: placed from the flame ({len(inl)}/{len(Xi)} sightings agree, median error {e:.1f} px)")
    calib["flame_3d"] = np.median(X, 0).tolist()
    return calib


def summary(calib, log=print):
    for k, c in calib["cameras"].items():
        if "R" not in c:
            log(f"  {k}: not calibrated")
            continue
        R, t = np.array(c["R"]), np.array(c["t"])
        cen = -R.T @ t
        log(f"  {k}: {c['method']:>5}  f={c['K'][0][0]:.0f}px  position {np.round(cen, 2)} m")


def save(calib, path):
    json.dump(calib, open(path, "w"), indent=1)


def load(path):
    return json.load(open(path))
