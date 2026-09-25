"""Build the reconstructed paradigm movie: frame i == paradigm frame i, at 25 fps.

Outputs (data/derived/paradigm/):
  paradigm_movie.mp4        125,743 frames, 25 fps, pts = p * 0.04; German (default) + English audio
  audio_de.m4a, audio_en.m4a  AAC 48 kHz stereo, t = paradigm time (for playback)
  audio_de_16k.flac, audio_en_16k.flac  mono 16 kHz, sample n <-> paradigm time n/16000,
                            i.e. paradigm frame p covers samples [640 p, 640 (p+1))  (for analysis)
  subs_de.srt/.vtt/.json, subs_en.*  subtitles re-timed to paradigm time

Audio re-timing: in the paradigm, film frames advance at 25 fps instead of the film's
23.976, so everything runs 25/23.976 = 4.27% faster. The audio is resampled by exactly
that ratio (like PAL playback without pitch correction: pitch rises by the same 4.27%),
then the head is trimmed and the removed section is cut out, using the same mapping as
the video.
"""

import json
import re
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import soxr

from . import paradigm
from .paradigm import FilmMapping
from .paths import PARADIGM

AUDIO_RATE = 48_000
ANALYSIS_RATE = 16_000
LANGS = {"de": ("deu", "Deutsch"), "en": ("eng", "English")}


# --- video ----------------------------------------------------------------------------------

def build_video(source: Path, frame_map: np.ndarray, out: Path, crf: int = 18, preset: str = "fast",
                limit: int | None = None):
    idx = frame_map["file_frame"]
    missing = frame_map["flag"] == 2
    n_out = limit or paradigm.N_FRAMES
    t_start = time.time()
    with av.open(str(source)) as inp, av.open(str(out), "w", options={"movflags": "+faststart"}) as o:
        vin = inp.streams.video[0]
        vin.thread_type = "AUTO"
        vs = o.add_stream("libx264", rate=paradigm.FPS)
        vs.width, vs.height = vin.codec_context.width, vin.codec_context.height
        vs.pix_fmt = "yuv420p"
        vs.time_base = Fraction(1, paradigm.FPS)
        # keyframe every second: fast, accurate seeking in the browser
        vs.options = {"crf": str(crf), "preset": preset, "g": str(paradigm.FPS)}
        decoded = inp.decode(vin)
        cur_i, cur = -1, None
        black = None
        for p in range(n_out):
            if missing[p]:
                if black is None:
                    black = av.VideoFrame.from_ndarray(
                        np.zeros((vs.height, vs.width, 3), np.uint8), format="rgb24").reformat(format="yuv420p")
                frame = black
            else:
                while cur_i < idx[p]:
                    cur = next(decoded)
                    cur_i += 1
                frame = cur
            f = frame.reformat(format="yuv420p")      # new frame object; source timing dropped below
            f.pts, f.time_base = p, Fraction(1, paradigm.FPS)
            for pkt in vs.encode(f):
                o.mux(pkt)
            if p % 2500 == 0:
                el = time.time() - t_start
                print(f"\r  video: {p:,}/{n_out:,}  {p / max(el, 1e-9):5.0f} fps  "
                      f"eta {(n_out - p) / max(p / max(el, 1e-9), 1):5.0f}s", end="", flush=True)
        for pkt in vs.encode():
            o.mux(pkt)
    print(f"\n  video done in {time.time() - t_start:.0f}s -> {out}")


# --- audio ----------------------------------------------------------------------------------

class _Sink:
    """Receives paradigm-time audio at 48 kHz and writes AAC (48k stereo) + FLAC (16k mono)."""

    def __init__(self, m4a: Path, flac: Path, channels: int, lang3: str, title: str):
        self.c_aac = av.open(str(m4a), "w")
        self.s_aac = self.c_aac.add_stream("aac", rate=AUDIO_RATE, layout="stereo" if channels == 2 else "mono")
        self.s_aac.bit_rate = 192_000
        self.s_aac.metadata.update(language=lang3, title=title)
        self.c_flac = av.open(str(flac), "w")
        self.s_flac = self.c_flac.add_stream("flac", rate=ANALYSIS_RATE, layout="mono")
        self.s_flac.format = "s16"
        self.fifo_aac, self.fifo_flac = av.AudioFifo(), av.AudioFifo()
        self.down = soxr.ResampleStream(AUDIO_RATE, ANALYSIS_RATE, 1, dtype="float32", quality="HQ")
        self.layout = "stereo" if channels == 2 else "mono"
        self.n_aac = self.n_flac = 0
        self.done = {}

    def _emit(self, fifo, stream, container, arr, layout, rate, fmt, frame_size):
        fr = av.AudioFrame.from_ndarray(arr, format=fmt, layout=layout)
        fr.sample_rate = rate
        fr.pts = None
        fifo.write(fr)
        while (chunk := fifo.read(frame_size)) is not None:
            self._encode(chunk, stream, container)

    def _encode(self, chunk, stream, container):
        n = self.done.get(stream, 0)
        chunk.pts, chunk.time_base = n, Fraction(1, chunk.sample_rate)   # sample-exact timestamps
        self.done[stream] = n + chunk.samples
        for pkt in stream.encode(chunk):
            container.mux(pkt)

    def write(self, x: np.ndarray, final=False):
        """x: (samples, channels) float32 at 48 kHz, paradigm time."""
        if len(x):
            self._emit(self.fifo_aac, self.s_aac, self.c_aac, np.ascontiguousarray(x.T, dtype=np.float32),
                       self.layout, AUDIO_RATE, "fltp", 1024)
            self.n_aac += len(x)
        mono = x.mean(axis=1, dtype=np.float32) if len(x) else np.zeros(0, np.float32)
        y = self.down.resample_chunk(mono, last=final)
        if len(y):
            s16 = (np.clip(y, -1, 1) * 32767).astype(np.int16)[None, :]
            self._emit(self.fifo_flac, self.s_flac, self.c_flac, s16, "mono", ANALYSIS_RATE, "s16", 4096)
            self.n_flac += len(y)

    def close(self):
        for fifo, s, c in ((self.fifo_aac, self.s_aac, self.c_aac), (self.fifo_flac, self.s_flac, self.c_flac)):
            rest = fifo.read()
            if rest is not None:
                self._encode(rest, s, c)
            for pkt in s.encode(None):
                c.mux(pkt)
            c.close()


def build_audio(source: Path, stream_index: int, mp: FilmMapping, lang: str):
    """Stream one language track through the mapping into paradigm time."""
    lang3, title = LANGS[lang]
    with av.open(str(source)) as inp:
        s = inp.streams[stream_index]
        in_rate, ch = s.codec_context.sample_rate, s.codec_context.channels
        assert float(s.start_time * s.time_base) == 0.0
        # paradigm-time segments [p0, p1) and where they start in the file (seconds)
        segs = [(0, min(mp.brk, paradigm.N_FRAMES))]
        if mp.brk < paradigm.N_FRAMES:
            segs.append((mp.brk, paradigm.N_FRAMES))
        speed = paradigm.FPS / mp.film_fps           # paradigm runs this much faster than the file
        sink = _Sink(PARADIGM / f"audio_{lang}.m4a", PARADIGM / f"audio_{lang}_16k.flac", ch, lang3, title)
        seg_state = []
        for p0, p1 in segs:
            t_file = float(mp.time(p0))            # file time of paradigm frame p0
            seg_state.append(dict(
                start=int(round(t_file * in_rate)),   # first input sample of the segment
                n_out=(p1 - p0) * AUDIO_RATE // paradigm.FPS,
                written=0,
                rs=soxr.ResampleStream(in_rate * speed, AUDIO_RATE, ch, dtype="float32", quality="VHQ")))
        pos = 0                                          # input sample index of the next decoded sample
        k = 0                                            # current segment
        for fr in inp.decode(s):
            x = fr.to_ndarray()                          # (ch, n) planar float
            x = x.T if x.shape[0] == ch else x.reshape(-1, ch)
            n = len(x)
            while k < len(seg_state) and n:
                st = seg_state[k]
                a = max(st["start"] - pos, 0)
                if a >= n:
                    break
                # an output segment ends when it has produced its exact sample count
                y = st["rs"].resample_chunk(np.ascontiguousarray(x[a:], dtype=np.float32))
                y = y[: st["n_out"] - st["written"]]
                sink.write(y)
                st["written"] += len(y)
                if st["written"] >= st["n_out"]:
                    k += 1                               # next segment picks up from its own start
                    continue
                break
            pos += n
        # flush + pad (the file may end before the paradigm does)
        for st in seg_state[k:]:
            y = st["rs"].resample_chunk(np.zeros((0, ch), np.float32), last=True)[: st["n_out"] - st["written"]]
            sink.write(y)
            st["written"] += len(y)
            if st["written"] < st["n_out"]:
                sink.write(np.zeros((st["n_out"] - st["written"], ch), np.float32))
                st["written"] = st["n_out"]
        sink.write(np.zeros((0, ch), np.float32), final=True)
        sink.close()
    total = sum(st["n_out"] for st in seg_state)
    print(f"  audio {lang}: {total / AUDIO_RATE:.2f}s at {AUDIO_RATE} Hz "
          f"(expected {paradigm.DURATION_S:.2f}s), speed factor {speed:.6f}")


# --- subtitles -------------------------------------------------------------------------------

def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    txt = path.read_text(encoding="utf-8-sig").replace("\r", "")
    ts = r"(\d+):(\d\d):(\d\d)[,.](\d{3})"
    cues = []
    for m in re.finditer(rf"{ts} --> {ts}[^\n]*\n(.*?)(?:\n\n|\Z)", txt, re.S):
        g = list(map(int, m.groups()[:8]))
        a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = re.sub(r"</?[^>]+>", "", m.group(9)).strip()
        if text:
            cues.append((a, b, text))
    return cues


def _fmt(t, sep):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def build_subtitles(mp: FilmMapping, src: Path, lang: str):
    cues = parse_srt(src)
    brk_t = mp.brk * paradigm.FRAME_S
    out = []
    for a, b, text in cues:
        ta, tb = mp.file_time_to_paradigm_time([a, b])
        if np.isnan(ta):                     # starts inside the removed section / before the head
            continue
        if np.isnan(tb) or (ta < brk_t <= tb):
            tb = min(brk_t, paradigm.DURATION_S) if ta < brk_t else paradigm.DURATION_S
        out.append((float(ta), float(tb), text))
    srt = "\n".join(f"{i + 1}\n{_fmt(a, ',')} --> {_fmt(b, ',')}\n{t}\n" for i, (a, b, t) in enumerate(out))
    vtt = "WEBVTT\n\n" + "\n".join(f"{_fmt(a, '.')} --> {_fmt(b, '.')}\n{t}\n" for a, b, t in out)
    (PARADIGM / f"subs_{lang}.srt").write_text(srt)
    (PARADIGM / f"subs_{lang}.vtt").write_text(vtt)
    (PARADIGM / f"subs_{lang}.json").write_text(json.dumps([dict(start=a, end=b, text=t) for a, b, t in out]))
    print(f"  subtitles {lang}: kept {len(out)}/{len(cues)} cues")


# --- mux --------------------------------------------------------------------------------------

def mux(video: Path, audios: list[tuple[Path, str, str]], out: Path):
    """Video stream of `video` + one AAC file per language -> `out` (may be the same file as `video`)."""
    tmp = out.with_suffix(".muxing.mp4")
    inputs = [av.open(str(video))] + [av.open(str(a)) for a, _, _ in audios]
    with av.open(str(tmp), "w", options={"movflags": "+faststart"}) as o:
        pairs = []
        vs = o.add_stream_from_template(inputs[0].streams.video[0])
        pairs.append((inputs[0].streams.video[0], vs))
        for i, (c, (_, lang3, title)) in enumerate(zip(inputs[1:], audios)):
            s = o.add_stream_from_template(c.streams.audio[0])
            s.metadata.update(language=lang3, title=title)
            s.disposition = av.stream.Disposition.default if i == 0 else av.stream.Disposition(0)
            pairs.append((c.streams.audio[0], s))
        # interleave by time
        its = [(c.demux(i), i, s) for c, (i, s) in zip(inputs, pairs)]
        heads = []
        for it, i, s in its:
            heads.append(next((p for p in it if p.dts is not None), None))
        while any(h is not None for h in heads):
            j = min((k for k, h in enumerate(heads) if h is not None),
                    key=lambda k: float(heads[k].dts * heads[k].time_base))
            pkt = heads[j]
            pkt.stream = its[j][2]
            o.mux(pkt)
            heads[j] = next((p for p in its[j][0] if p.dts is not None), None)
    for c in inputs:
        c.close()
    tmp.replace(out)
    print(f"  muxed -> {out}")
