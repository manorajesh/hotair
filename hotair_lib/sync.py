"""Sync cameras from their audio. Works with a clap, but also with ordinary room noise (shuffling, footsteps).

Result: offset[cam] such that  time_in_cam_file = time_on_reference_clock + offset[cam].
"""
import numpy as np
from scipy.signal import fftconvolve

from .media import audio

SR = 16000


def _env(a, hop):
    a = a - np.convolve(a, np.ones(16) / 16, "same")          # drop hum and rumble
    e = np.abs(a)
    e = np.convolve(e, np.ones(2 * hop) / (2 * hop), "same")[::hop]
    e = np.log1p(e)
    rate = SR / hop
    e = e - np.convolve(e, np.ones(int(rate * 0.2)) / int(rate * 0.2), "same")   # onsets, not loudness
    return (e - e.mean()) / (e.std() + 1e-9), rate


def offset(ref_audio, other_audio, search=None):
    """Seconds to add to a reference time to get the same moment in `other`."""
    # coarse: 200 Hz envelope, whole clip
    er, rate = _env(ref_audio, 80)
    eo, _ = _env(other_audio, 80)
    c = fftconvolve(eo, er[::-1], mode="full")
    lags = (np.arange(len(c)) - (len(er) - 1)) / rate
    if search is not None:
        c = np.where(np.abs(lags - search[0]) <= search[1], c, -np.inf)
    i = int(np.argmax(c))
    coarse = lags[i]
    excl = np.abs(lags - coarse) > 0.25
    confidence = float(c[i] / (np.max(np.where(excl, c, -np.inf)) + 1e-9))
    # fine: 2 kHz envelope near the coarse answer, with sub-sample peak
    er, rate = _env(ref_audio, 8)
    eo, _ = _env(other_audio, 8)
    c = fftconvolve(eo, er[::-1], mode="full")
    lags = (np.arange(len(c)) - (len(er) - 1)) / rate
    c = np.where(np.abs(lags - coarse) < 0.2, c, -np.inf)
    i = int(np.argmax(c))
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if np.isfinite(y0 + y2) and (y0 - 2 * y1 + y2) != 0 else 0.0
    return float(lags[i] + d / rate), confidence


def sync_files(files, ref_cam, log=print):
    """files: {cam: path}. Returns {cam: (offset_s, confidence)}."""
    A = {k: audio(p, SR) for k, p in files.items()}
    for k, a in A.items():
        if a.size == 0:
            raise RuntimeError(f"{files[k]} has no audio track, so it can't be synced from sound.")
    res = {ref_cam: (0.0, float("inf"))}
    for k in files:
        if k == ref_cam:
            continue
        off, conf = offset(A[ref_cam], A[k])
        res[k] = (off, conf)
        flag = "" if conf > 2.0 else "   <-- weak match: check this camera (clap next time)"
        log(f"  {k}: {off:+.4f} s vs {ref_cam}  (match strength {conf:.1f}x){flag}")
    return res
