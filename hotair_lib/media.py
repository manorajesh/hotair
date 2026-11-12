"""Video/audio input and output through ffmpeg. All times are seconds in each file's own clock."""
import json
import shutil
import subprocess
import sys

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
HWACCEL = ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []
_DRAWTEXT = None


def has_drawtext():
    """Homebrew's default ffmpeg is built without freetype, so the drawtext filter may be missing."""
    global _DRAWTEXT
    if _DRAWTEXT is None:
        r = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True)
        _DRAWTEXT = any(line.split()[1:2] == ["drawtext"] for line in r.stdout.splitlines())
        if not _DRAWTEXT:
            print("  note: this ffmpeg has no drawtext filter; grid videos will have no camera labels", file=sys.stderr)
    return _DRAWTEXT


def probe(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,width,height,avg_frame_rate:stream_side_data=rotation",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s.get("codec_type") == "video")
    w, h = v["width"], v["height"]
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        rot = int(sd.get("rotation", 0) or 0)
    if abs(rot) in (90, 270):
        w, h = h, w
    num, den = (v.get("avg_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1)
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    return dict(width=w, height=h, fps=fps, duration=float(info["format"]["duration"]), audio=has_audio)


def decode(path, start, dur, fps, width=None, color=False):
    """Yield frames from `start` for `dur` seconds at a constant `fps`, optionally resized to `width`."""
    info = probe(path)
    if width:
        w = int(width)
        h = int(round(info["height"] * w / info["width"] / 2) * 2)
    else:
        w, h = info["width"], info["height"]
    vf = f"fps={fps}" + (f",scale={w}:{h}:flags=area" if width else "")
    pix, ch = ("bgr24", 3) if color else ("gray", 1)
    cmd = [FFMPEG, "-v", "error"] + HWACCEL + ["-ss", f"{max(start, 0):.4f}", "-i", path, "-t", f"{dur:.4f}",
                                               "-vf", vf, "-pix_fmt", pix, "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * ch
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            a = np.frombuffer(buf, np.uint8)
            yield a.reshape(h, w, 3) if color else a.reshape(h, w)
    finally:
        p.stdout.close()
        p.wait()


def audio(path, sr=16000):
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32)


class Writer:
    """H.264 mp4 writer fed with numpy frames (grey or BGR)."""

    def __init__(self, path, w, h, fps, color=True, crf=18):
        self.color = color
        self.p = subprocess.Popen([FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24" if color else "gray",
                                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                   "-crf", str(crf), "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def grid_video(inputs, out, labels=None, cols=None, tile_w=640):
    """Stack already-synced mp4 files into one grid video with ffmpeg."""
    n = len(inputs)
    cols = cols or (3 if n > 4 else 2 if n > 1 else 1)
    rows = int(np.ceil(n / cols))
    args = []
    for f in inputs:
        args += ["-i", f]
    chains = []
    for i in range(n):
        lab = ""
        if labels and has_drawtext():
            lab = f",drawtext=text='{labels[i]}':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5"
        chains.append(f"[{i}:v]scale={tile_w}:-2,setsar=1{lab}[v{i}]")
    pad = []
    i0 = probe(inputs[0])
    tile_h = int(round(i0["height"] * tile_w / i0["width"] / 2) * 2)
    for i in range(n, rows * cols):
        pad.append(f"color=c=black:s={tile_w}x{tile_h}[v{i}]")
    layout = "|".join(("+".join([f"w{j}" for j in range(i % cols)]) or "0") + "_" +
                      ("+".join([f"h{j * cols}" for j in range(i // cols)]) or "0") for i in range(rows * cols))
    fc = ";".join(chains + pad) + ";" + "".join(f"[v{i}]" for i in range(rows * cols)) + \
        f"xstack=inputs={rows * cols}:layout={layout}:fill=black:shortest=1[out]"
    subprocess.run([FFMPEG, "-y", "-v", "error"] + args + ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264",
                                                           "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", out],
                   check=True)
