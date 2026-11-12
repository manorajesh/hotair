#!/usr/bin/env python3
"""
Hot Air: single-camera background-oriented schlieren (BOS) processor.

Turns a video of you breathing in front of a speckle board into a video of the air.

  python bos.py MVI_0001.MOV                      # reference = first 2 s
  python bos.py MVI_0001.MOV --ref 1.0 3.0        # reference = seconds 1.0 to 3.0 (you holding your breath)
  python bos.py MVI_0001.MOV --start 3 --end 20   # only process seconds 3 to 20
  python bos.py MVI_0001.MOV --crop 600 200 900 600   # x y width height, in pixels

Outputs (next to the input file):
  <name>_schlieren.mp4   grey "knife-edge" look: sideways shift as light/dark
  <name>_magnitude.mp4   how much the air bent light, black = nothing, bright = strong
  <name>_side.mp4        original | schlieren, side by side
  <name>_still.png       the single strongest frame

Needs: pip install opencv-python numpy   (and ffmpeg on your PATH for the mp4 files)
"""
import argparse, os, subprocess, sys
import numpy as np
import cv2


def parse():
    p = argparse.ArgumentParser(description="Single-camera BOS: speckle video -> image of the air")
    p.add_argument("video")
    p.add_argument("--ref", nargs=2, type=float, default=[0.0, 2.0], metavar=("FROM", "TO"),
                   help="seconds with NO breath, used as the reference (default 0 2)")
    p.add_argument("--start", type=float, default=None, help="first second to process (default: end of reference)")
    p.add_argument("--end", type=float, default=None, help="last second to process (default: end of clip)")
    p.add_argument("--crop", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"))
    p.add_argument("--smooth", type=float, default=2.0, help="spatial blur of the result in pixels (default 2)")
    p.add_argument("--temporal", type=int, default=2, help="average this many frames together (default 2, 1 = off)")
    p.add_argument("--gain", type=float, default=None,
                   help="shift in pixels that maps to full black/white (default: automatic)")
    p.add_argument("--quality", choices=["fast", "medium", "fine"], default="medium",
                   help="optical-flow quality (default medium)")
    return p.parse_args()


def gray(frame, crop):
    if crop:
        x, y, w, h = crop
        frame = frame[y:y + h, x:x + w]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def read_range(cap, fps, t0, t1, crop):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
    out = []
    for _ in range(max(1, int(round((t1 - t0) * fps)))):
        ok, f = cap.read()
        if not ok:
            break
        out.append(gray(f, crop))
    return out


def make_flow(quality):
    preset = {"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
              "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
              "fine": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[quality]
    dis = cv2.DISOpticalFlow_create(preset)
    if quality == "fine":
        dis.setFinestScale(0)
        dis.setPatchStride(2)
        dis.setVariationalRefinementIterations(10)
    return dis


def shift(dis, ref, cur, smooth):
    flow = dis.calc(ref, cur, None)
    # remove whole-frame drift (tripod nudges, floor vibration)
    flow -= np.median(flow.reshape(-1, 2), axis=0)
    if smooth > 0:
        flow = cv2.GaussianBlur(flow, (0, 0), smooth)
    return flow


class Writer:
    def __init__(self, path, w, h, fps):
        self.path = path
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", path]
        try:
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.cv = None
        except FileNotFoundError:
            print("ffmpeg not found, writing with OpenCV instead (bigger files)")
            self.p = None
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.p:
            self.p.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.cv.write(img)

    def close(self):
        if self.p:
            self.p.stdin.close()
            self.p.wait()
        else:
            self.cv.release()


def main():
    a = parse()
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"Could not open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    start = a.start if a.start is not None else a.ref[1]
    end = min(a.end if a.end is not None else dur, dur)
    print(f"{a.video}: {dur:.1f} s at {fps:.2f} fps; reference {a.ref[0]}-{a.ref[1]} s; processing {start:.1f}-{end:.1f} s")

    # 1. reference = median of the still frames (cuts sensor and compression noise)
    refs = read_range(cap, fps, a.ref[0], a.ref[1], a.crop)
    if not refs:
        sys.exit("No reference frames read; check --ref")
    ref = np.median(np.stack(refs), axis=0).astype(np.uint8)
    h, w = ref.shape
    dis = make_flow(a.quality)

    # 2. automatic contrast: sample frames across the clip
    gain = a.gain
    if gain is None:
        samples = []
        for t in np.linspace(start, end, 12, endpoint=False):
            fr = read_range(cap, fps, t, t + 1 / fps, a.crop)
            if fr:
                samples.append(np.abs(shift(dis, ref, fr[0], a.smooth)))
        gain = float(np.percentile(np.stack(samples), 99.7)) if samples else 1.0
        gain = max(gain, 0.05)
    print(f"Contrast: {gain:.2f} px of shift = full black/white")

    base = os.path.splitext(a.video)[0]
    out_fps = fps
    ws = Writer(base + "_schlieren.mp4", w, h, out_fps)
    wm = Writer(base + "_magnitude.mp4", w, h, out_fps)
    wb = Writer(base + "_side.mp4", w * 2, h, out_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    n_total = int((end - start) * fps)
    window, best, best_score = [], None, -1
    for i in range(n_total):
        ok, frame = cap.read()
        if not ok:
            break
        g = gray(frame, a.crop)
        window.append(shift(dis, ref, g, a.smooth))
        window = window[-a.temporal:]
        flow = np.mean(window, axis=0)

        dx, dy = flow[..., 0], flow[..., 1]
        schl = np.clip(128 + 127 * dx / gain, 0, 255).astype(np.uint8)
        mag = np.hypot(dx, dy)
        magi = np.clip(255 * (mag / (gain * 1.2)) ** 0.8, 0, 255).astype(np.uint8)

        ws.write(schl)
        wm.write(magi)
        wb.write(np.hstack([cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), cv2.cvtColor(schl, cv2.COLOR_GRAY2BGR)]))

        score = float(np.percentile(mag, 99))
        if score > best_score:
            best_score, best = score, schl.copy()
        if i % 30 == 0:
            print(f"  {i}/{n_total} frames", end="\r")

    for wr in (ws, wm, wb):
        wr.close()
    if best is not None:
        cv2.imwrite(base + "_still.png", best)
    print(f"\nDone: {base}_schlieren.mp4, _magnitude.mp4, _side.mp4, _still.png")


if __name__ == "__main__":
    main()
