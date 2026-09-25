"""Reading a movie file: probing, fingerprinting, per-frame signals, frames and audio."""

import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image

from .paths import CACHE


def probe(path: Path) -> dict:
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        return dict(
            file=str(path),
            bytes=path.stat().st_size,
            duration_s=c.duration / 1e6,
            video=dict(codec=v.codec_context.name, width=v.codec_context.width,
                       height=v.codec_context.height, avg_rate=str(v.average_rate),
                       frames=v.frames, start_s=float(v.start_time * v.time_base)),
            audio=[dict(index=s.index, codec=s.codec_context.name, rate=s.codec_context.sample_rate,
                        channels=s.codec_context.channels, language=s.metadata.get("language"))
                   for s in c.streams.audio],
        )


def fingerprint(path: Path) -> str:
    """Hash of the compressed video packets: identifies the *content*, not the container.
    Cached by (path, size, mtime) so repeated checks are instant."""
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    cache_file = CACHE / "fingerprints.json"
    try:
        cache = json.loads(cache_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    if key not in cache:
        h = hashlib.sha1()
        with av.open(str(path)) as c:
            for pkt in c.demux(c.streams.video[0]):
                if pkt.size:
                    h.update(bytes(pkt))
        cache[key] = h.hexdigest()[:16]
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1))
    return cache[key]


LANG3 = {"de": ("deu", "ger"), "en": ("eng",)}


def audio_stream_index(path: Path, stream: int | None, lang: str) -> int:
    """Stream index for a configured audio source: explicit, or the stream tagged with `lang`."""
    with av.open(str(path)) as c:
        audio = {s.index: s.metadata.get("language") for s in c.streams.audio}
    if stream is not None:
        if stream not in audio:
            raise SystemExit(f"{path.name}: stream #{stream} is not an audio stream (audio: {audio}). "
                             f"Run `summer probe {path}`.")
        return stream
    tagged = [i for i, l in audio.items() if l in LANG3[lang]]
    if len(tagged) == 1:
        return tagged[0]
    if len(audio) == 1 and lang == "de":
        return next(iter(audio))       # single untagged track: assume it is the German one, as configured
    raise SystemExit(f"{path.name}: can't tell which audio stream is '{lang}' (streams: {audio}). "
                     f"Set it explicitly in summer.toml, e.g. {lang} = \"movie/{path.name}#1\"; "
                     f"`summer probe {path}` lists the streams.")


def frame_signal(path: Path, fp: str, w: int = 96, h: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """(pts_s, mad) for every decoded frame; mad[i] = mean |frame i - frame i-1| on a small
    greyscale thumbnail. Hard cuts are spikes; duplicated frames are ~0. Cached per fingerprint."""
    cache = CACHE / f"signal_{fp}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["pts"], z["mad"]
    t_start = time.time()
    pts, mad, prev = [], [], None
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            g = fr.reformat(width=w, height=h, format="gray").to_ndarray().astype(np.int16)
            mad.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
            pts.append(float(fr.pts * fr.time_base))
            prev = g
            if i % 10_000 == 0:
                print(f"\r  decoding frames: {i:,}  ({time.time() - t_start:.0f}s)", end="", flush=True)
    print()
    pts, mad = np.array(pts), np.array(mad, dtype=np.float32)
    assert np.all(np.diff(pts) > 0), "decoded frames are not in presentation order"
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pts=pts, mad=mad)
    return pts, mad


def nearest_frame(frame_pts: np.ndarray, t) -> np.ndarray:
    """Index of the decoded frame whose pts is nearest to t."""
    t = np.asarray(t, dtype=np.float64)
    j = np.clip(np.searchsorted(frame_pts, t), 1, len(frame_pts) - 1)
    return np.where(np.abs(frame_pts[j - 1] - t) <= np.abs(frame_pts[j] - t), j - 1, j)


def grab_frames(path: Path, frame_pts: np.ndarray, indices, width: int | None = None) -> dict[int, Image.Image]:
    """Decode specific frames (by decode index) exactly, seeking to the preceding keyframe."""
    want = sorted(set(int(i) for i in indices))
    out = {}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        tb = s.time_base
        for idx in want:
            if idx in out:
                continue
            target = frame_pts[idx]
            c.seek(int((target - 5) / tb) if target > 5 else 0, stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                t = float(fr.pts * tb)
                if abs(t - target) < 1e-4:
                    im = fr.to_image()
                    if width:
                        im = im.resize((width, round(im.height * width / im.width)))
                    out[idx] = im
                    break
                if t > target + 1:
                    raise RuntimeError(f"could not decode frame {idx} at {target:.3f}s")
    return out


def decode_audio(path: Path, stream_index: int, rate: int | None = None, mono: bool = False):
    """Whole audio track as float32 (channels, samples), optionally resampled. Starts at t=0."""
    with av.open(str(path)) as c:
        s = c.streams[stream_index]
        assert float(s.start_time * s.time_base) == 0.0, "audio stream does not start at t=0"
        rs = av.AudioResampler(format="fltp", layout="mono" if mono else s.codec_context.layout.name,
                               rate=rate or s.codec_context.sample_rate)
        chunks = []
        for fr in c.decode(s):
            chunks += [o.to_ndarray() for o in rs.resample(fr)]
        chunks += [o.to_ndarray() for o in rs.resample(None)]
    return np.concatenate(chunks, axis=1), rate or s.codec_context.sample_rate
