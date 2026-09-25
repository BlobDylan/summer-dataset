"""Checks that a movie file + mapping reproduce the paradigm, with controls, and an HTML report.

Used twice: on the source file with the fitted mapping, and on the built paradigm movie
with the identity mapping (round trip: the build must preserve the alignment).
"""

import base64
import html
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import webrtcvad
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d

from . import movie, nwb, paradigm
from .paradigm import FilmMapping

# pass criteria
MIN_EXACT_CUT_FRACTION = 0.80     # per segment, among all annotated cuts
MAX_CONTROL_EXACT_FRACTION = 0.10 # same statistic with the head offset shifted by one frame
MAX_SPEECH_OFFSET_EDGE_S = 0.15   # annotated speech offsets vs detected speech offsets


def cut_offsets(mp: FilmMapping, frame_pts, mad, frames, window=5):
    """For each paradigm frame, where (relative to the mapped file frame) is the biggest
    frame difference within +-window frames. 0 = the transition is exactly where predicted."""
    idx = movie.nearest_frame(frame_pts, mp.time(frames))
    w = np.arange(-window, window + 1)
    m = mad[np.clip(idx[:, None] + w[None, :], 0, len(mad) - 1)]
    return w[m.argmax(1)]


def check_cuts(mp: FilmMapping, frame_pts, mad) -> dict:
    cuts = nwb.onsets("camera-cuts")
    cuts = cuts[mp.time(cuts) < frame_pts[-1]]
    segs = {"A": cuts < mp.brk, "B": cuts >= mp.brk}

    def stats(m):
        out = {}
        for name, sel in segs.items():
            if sel.sum() == 0:
                continue
            o = cut_offsets(m, frame_pts, mad, cuts[sel])
            out[name] = dict(n=int(sel.sum()), exact=float(np.mean(o == 0)),
                             within_1=float(np.mean(np.abs(o) <= 1)))
        return out

    res = {"mapping": stats(mp),
           "control_head_minus_1": stats(replace(mp, head=mp.head - 1)),
           "control_head_plus_1": stats(replace(mp, head=mp.head + 1))}
    if mp.gap:
        res["control_no_gap"] = stats(replace(mp, gap=0))
    rng = np.random.default_rng(0)
    rnd = np.sort(rng.integers(paradigm.ANALYSIS_FIRST, paradigm.ANALYSIS_LAST, len(cuts)))
    res["control_random_frames_exact"] = float(np.mean(cut_offsets(mp, frame_pts, mad, rnd) == 0))
    days = nwb.onsets("days-of-summer")
    res["days_of_summer_onsets"] = dict(n=int(len(days)),
                                        exact=float(np.mean(cut_offsets(mp, frame_pts, mad, days) == 0)))
    ok = all(v["exact"] >= MIN_EXACT_CUT_FRACTION for v in res["mapping"].values())
    ok &= all(v["exact"] <= MAX_CONTROL_EXACT_FRACTION
              for k in ("control_head_minus_1", "control_head_plus_1") for v in res[k].values())
    res["pass"] = bool(ok)
    return res


def vad_track(source: Path, stream_index: int) -> np.ndarray:
    """10 ms voice-activity decisions for an audio track (in the file's clock)."""
    a, _ = movie.decode_audio(source, stream_index, rate=16000, mono=True)
    a = (np.clip(a[0], -1, 1) * 32767).astype(np.int16)
    v, n = webrtcvad.Vad(3), len(a) // 160
    frames = a[: n * 160].reshape(n, 160)
    return np.fromiter((v.is_speech(f.tobytes(), 16000) for f in frames), np.uint8, n)


def check_audio(mp: FilmMapping, vad: np.ndarray) -> dict:
    """Correlate detected speech with the annotated main-character speech, and compare edges."""
    ind = nwb.indicator_functions()
    speak = ((ind["tom-speaking"] + ind["summer-speaking"]) > 0).astype(float)
    P = np.arange(paradigm.ANALYSIS_FIRST, paradigm.ANALYSIS_LAST + 1)
    env = uniform_filter1d(vad.astype(float), 5)
    tt = (np.arange(len(env)) + 0.5) * 0.01

    def corr(m, lag=0.0):
        x = np.interp(m.paradigm_time_to_file_time((P + 0.5) * paradigm.FRAME_S) + lag, tt, env)
        y = speak[P]
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))

    lags = np.round(np.arange(-1.5, 1.501, 0.02), 2)
    r = np.array([corr(mp, lag) for lag in lags])
    # control: same mapping but the wrong speed (off by the PAL factor 25/23.976), same start point
    wrong_fps = 24000 / 1001 if mp.film_fps == paradigm.FPS else paradigm.FPS
    wrong = replace(mp, film_fps=wrong_fps, t0=mp.t0 + mp.head * (1 / mp.film_fps - 1 / wrong_fps))
    r_ctrl = max(corr(wrong, lag) for lag in lags)

    # edges: annotated on/offsets vs nearest detected speech on/offsets
    s = binary_opening(binary_closing(vad > 0, np.ones(15)), np.ones(10)).astype(np.int8)
    d = np.diff(np.r_[0, s, 0])
    v_on, v_off = np.flatnonzero(d == 1) * 0.01, np.flatnonzero(d == -1) * 0.01
    iv = np.concatenate([nwb.intervals("tom-speaking"), nwb.intervals("summer-speaking")])

    def edge(label_frames, detected):
        t = mp.paradigm_time_to_file_time(label_frames * paradigm.FRAME_S)
        j = np.clip(np.searchsorted(detected, t), 1, len(detected) - 1)
        a, b = detected[j] - t, detected[j - 1] - t
        dd = np.where(np.abs(a) < np.abs(b), a, b)
        return float(np.median(dd[np.abs(dd) < 1]))

    res = dict(r_at_zero_lag=corr(mp), best_lag_s=float(lags[r.argmax()]), r_best=float(r.max()),
               control_wrong_speed_best_r=r_ctrl,
               median_onset_shift_s=edge(iv[:, 0], v_on),
               median_offset_shift_s=edge(iv[:, 1], v_off))
    res["pass"] = bool(res["r_at_zero_lag"] > 3 * max(res["control_wrong_speed_best_r"], 0.01)
                       and abs(res["median_offset_shift_s"]) <= MAX_SPEECH_OFFSET_EDGE_S)
    return res


# --- report ---------------------------------------------------------------------------------

def _img(im) -> str:
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _sheet(title, note, items) -> str:
    cells = "".join(f'<figure><img src="{src}"><figcaption>{html.escape(cap)}</figcaption></figure>'
                    for src, cap in items)
    return f"<h2>{html.escape(title)}</h2><p>{html.escape(note)}</p><div class=grid>{cells}</div>"


def contact_sheets(source: Path, mp: FilmMapping, frame_pts) -> str:
    ind = nwb.indicator_functions()
    labels = sorted(ind)

    def active(p):
        return ", ".join(n for n in labels if ind[n][p] and n not in ("indoor-setting", "persons"))

    def grab(ps, width=320):
        idx = movie.nearest_frame(frame_pts, mp.time(np.array(ps)))
        ims = movie.grab_frames(source, frame_pts, idx, width=width)
        return [(_img(ims[int(i)]), f"p={p}  |  {active(p)}") for p, i in zip(ps, idx)]

    out = []
    days = nwb.intervals("days-of-summer")
    out.append(_sheet("Days-of-Summer title cards",
                      "Paradigm frame 12 frames after each annotated title-card onset. All should be day-number cards.",
                      grab([int(a) + 12 for a, _ in days])))
    if mp.gap:
        b = mp.brk
        out.append(_sheet("Around the break",
                          f"Paradigm frames {b - 3}..{b + 2}; the removed section sits between {b - 1} and {b}.",
                          grab(list(range(b - 3, b + 3)))))
    rng = np.random.default_rng(1)
    faces = np.flatnonzero((ind["summer-faces"] | ind["tom-faces"]).astype(bool))
    faces = faces[(faces > paradigm.ANALYSIS_FIRST) & (faces < paradigm.ANALYSIS_LAST)]
    out.append(_sheet("Random frames with a main-character face annotated",
                      "Check that the listed people are visible.", grab(sorted(rng.choice(faces, 12, replace=False)))))
    cuts = nwb.onsets("camera-cuts")
    pick = sorted(rng.choice(cuts[(cuts > 1000) & (cuts < 119000)], 6, replace=False))
    pairs = []
    for c in pick:
        pairs += grab([int(c) - 1, int(c)], width=240)
    out.append(_sheet("Annotated cuts: last frame of old shot / first frame of new shot",
                      "Each pair should straddle a shot change exactly.", pairs))
    return "".join(out)


def run(source: Path, mp: FilmMapping, audio_stream: int | None, out_dir: Path, title: str,
        audio_source: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = movie.fingerprint(source)
    frame_pts, mad = movie.frame_signal(source, fp)
    res = {"source": str(source), "fingerprint": fp, "mapping": mp.to_dict(),
           "cuts": check_cuts(mp, frame_pts, mad)}
    if audio_stream is not None:
        res["audio"] = check_audio(mp, vad_track(audio_source or source, audio_stream))
    res["pass"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v)
    (out_dir / "verification.json").write_text(json.dumps(res, indent=2))
    sheets = contact_sheets(source, mp, frame_pts)
    page = f"""<!doctype html><meta charset=utf-8><title>Alignment report</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1400px;color:#222;background:#fff}}
pre{{background:#f4f4f4;padding:12px;overflow:auto}}.grid{{display:flex;flex-wrap:wrap;gap:8px}}
figure{{margin:0;width:320px}}figure img{{width:100%}}figcaption{{font-size:11px}}
.ok{{color:#070}}.bad{{color:#b00}}</style>
<h1>{html.escape(title)}</h1>
<p>Overall: <b class={"ok" if res["pass"] else "bad"}>{"PASS" if res["pass"] else "FAIL"}</b></p>
<pre>{html.escape(json.dumps(res, indent=2))}</pre>{sheets}"""
    (out_dir / "report.html").write_text(page)
    print(f"{title}: {'PASS' if res['pass'] else 'FAIL'}  ->  {out_dir / 'report.html'}")
    return res
