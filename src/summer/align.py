"""Measure how a movie file maps onto paradigm frames, using the annotated camera cuts.

Idea: the 958 annotated `camera-cuts` are exact paradigm frames where a new shot starts.
Detect hard cuts in the file (spikes in frame difference), convert their times to
film-frame positions for a candidate film rate, and histogram (film position - paradigm
frame) over all pairs. The right film rate produces sharp integer peaks: one per
contiguous segment (the head offset, and head + the removed section). See paradigm.FilmMapping.
"""

import json
from pathlib import Path

import numpy as np

from . import movie, nwb, paradigm
from .paradigm import FilmMapping
from .paths import ALIGNMENT

CANDIDATE_FILM_FPS = {"25 (PAL)": 25.0, "23.976 (film)": 24000 / 1001, "24": 24.0}


def detect_cuts(mad: np.ndarray) -> np.ndarray:
    """Frames that start a new shot: a big difference that dominates both neighbours."""
    nb = np.maximum(np.r_[0, mad[:-1]], np.r_[mad[1:], 0])
    return np.flatnonzero((mad > 8) & (mad > 3 * nb))


def transition_score(mad: np.ndarray) -> np.ndarray:
    """Frame difference as a percentile rank (0..1): robust to dark/bright scenes."""
    r = np.empty(len(mad))
    r[np.argsort(mad, kind="stable")] = np.arange(len(mad))
    return r / len(mad)


def _offset_hist(det_pos: np.ndarray, cuts: np.ndarray, lo=-3000, hi=30000):
    d = np.round(det_pos[None, :] - cuts[:, None]).astype(np.int64).ravel()
    d = d[(d >= lo) & (d < hi)]
    return np.bincount(d - lo, minlength=hi - lo), lo


def _peaks(h, lo, n=4, guard=50):
    h = h.astype(float).copy()
    out = []
    for _ in range(n):
        j = int(h.argmax())
        out.append((j + lo, int(h[j])))
        h[max(0, j - guard): j + guard] = -1
    return out


def fit(frame_pts: np.ndarray, mad: np.ndarray) -> tuple[FilmMapping, dict]:
    cuts = nwb.onsets("camera-cuts")
    det_t = frame_pts[detect_cuts(mad)]
    diag = {"n_annotated_cuts": int(len(cuts)), "n_detected_cuts": int(len(det_t)), "film_fps_scan": {}}

    # 1. film rate: the one whose offset histogram has the sharpest peak
    best = None
    for name, F in CANDIDATE_FILM_FPS.items():
        h, lo = _offset_hist(det_t * F, cuts)
        pk = _peaks(h, lo)
        diag["film_fps_scan"][name] = pk
        if best is None or pk[0][1] > best[2][0][1]:
            best = (name, F, pk)
    name, F, pk = best

    # 2. segments: peaks well above the background, ordered by where their cuts sit
    background = np.median([c for _, c in pk[2:]]) if len(pk) > 2 else 0
    offsets = [o for o, c in pk if c > max(20, 5 * background)]
    det_pos = det_t * F

    def matched(off):
        pos = cuts + off
        j = np.clip(np.searchsorted(det_pos, pos), 1, len(det_pos) - 1)
        dist = np.minimum(np.abs(det_pos[j] - pos), np.abs(det_pos[j - 1] - pos))
        return cuts[dist <= 1.0]

    segs = sorted(((matched(o), o) for o in offsets), key=lambda s: np.median(s[0]))
    assert 1 <= len(segs) <= 2, f"expected 1 or 2 segments, found offsets {offsets}"
    head = segs[0][1]
    diag["segments"] = [dict(offset=int(o), n_matched=int(len(m)), first=int(m.min()), last=int(m.max()))
                        for m, o in segs]

    # 3. sub-frame clock offset t0 (median residual of matched cuts)
    m0 = segs[0][0]
    j = np.clip(np.searchsorted(det_pos, m0 + head), 1, len(det_pos) - 1)
    near = np.where(np.abs(det_pos[j] - (m0 + head)) < np.abs(det_pos[j - 1] - (m0 + head)), det_pos[j], det_pos[j - 1])
    t0 = float(np.median(near - (m0 + head)) / F)

    if len(segs) == 1:
        return FilmMapping(F, int(head), paradigm.N_FRAMES, 0, t0), diag

    # 4. the break: somewhere between the last cut of segment A and the first of segment B.
    # Pick the frame where annotated transitions best land on transitions in the file.
    (ma, _), (mb, off_b) = segs
    # a few cuts match the other offset by coincidence; split where the two sets separate best
    splits = np.unique(np.r_[ma, mb])
    agree = [(ma < s).sum() + (mb >= s).sum() for s in splits]
    s = splits[int(np.argmax(agree))]
    diag["segment_split"] = dict(at=int(s), coincidental_matches=int(len(ma) + len(mb) - max(agree)))
    assert diag["segment_split"]["coincidental_matches"] < 0.05 * (len(ma) + len(mb)), "no clean split"
    ma, mb = ma[ma < s], mb[mb >= s]
    gap = int(off_b - head)
    score = transition_score(mad)
    trans = np.unique(np.r_[cuts, nwb.onsets("scenes"), nwb.onsets("days-of-summer")])
    # the first B match may itself be coincidental, so search a little past it
    cands = np.arange(ma.max() + 1, mb.min() + 60)
    near_trans = trans[(trans >= cands[0] - 50) & (trans <= cands[-1] + 50)]
    scores = []
    for b in cands:
        mp = FilmMapping(F, int(head), int(b), gap, t0)
        scores.append(score[movie.nearest_frame(frame_pts, mp.time(near_trans))].sum())
    scores = np.array(scores)
    brk = int(cands[np.flatnonzero(scores == scores.max())[-1]])   # ties: keep more frames in segment A
    diag["break_search"] = dict(range=[int(cands[0]), int(cands[-1])],
                                transitions_used=near_trans.tolist(),
                                best=brk, n_tied=int((scores == scores.max()).sum()),
                                published_wrapper_brk=97212,
                                score_at_published=float(scores[cands == 97212][0]) if 97212 in cands else None,
                                score_at_best=float(scores.max()))
    return FilmMapping(F, int(head), brk, gap, t0), diag


def frame_map(mp: FilmMapping, frame_pts: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """Per paradigm frame: file frame index, file time, and a quality flag.

    flag 0: exact; 1: within a duplicate/drop burst of a re-timed file (may be off by one
    frame, only where there is motion; in static shots the neighbours look the same);
    2: beyond the end of the file (no picture).
    """
    p = np.arange(paradigm.N_FRAMES)
    t = mp.time(p)
    idx = movie.nearest_frame(frame_pts, t)
    dt = np.median(np.diff(frame_pts))
    flag = np.zeros(len(p), np.uint8)
    # duplicated frames that are visible (in motion) mark the bursts
    prev, nxt = np.r_[0, mad[:-1]], np.r_[mad[1:], 0]
    dups = np.flatnonzero((mad < 0.15 * np.minimum(prev, nxt)) & (np.minimum(prev, nxt) > 1.5))
    burst = np.zeros(len(frame_pts), bool)
    for d in dups:
        burst[max(0, d - 8): d + 9] = True
    flag[burst[idx]] = 1
    flag[t > frame_pts[-1] + dt / 2] = 2
    out = np.zeros(paradigm.N_FRAMES, dtype=[("file_frame", "i4"), ("file_time", "f8"), ("flag", "u1")])
    out["file_frame"], out["file_time"], out["flag"] = idx, t, flag
    return out


def run(source: Path) -> dict:
    ALIGNMENT.mkdir(parents=True, exist_ok=True)
    info = movie.probe(source)
    print(f"source: {source.name}  {info['video']['width']}x{info['video']['height']} "
          f"{info['video']['avg_rate']} fps, {info['video']['frames']:,} frames")
    fp = movie.fingerprint(source)
    frame_pts, mad = movie.frame_signal(source, fp)
    mp, diag = fit(frame_pts, mad)
    fm = frame_map(mp, frame_pts, mad)
    np.save(ALIGNMENT / "frame_map.npy", fm)
    result = dict(source=info | {"fingerprint": fp}, mapping=mp.to_dict(), fit=diag,
                  frame_map_flags={"exact": int((fm["flag"] == 0).sum()),
                                   "burst_maybe_off_by_one": int((fm["flag"] == 1).sum()),
                                   "beyond_file_end": int((fm["flag"] == 2).sum())})
    (ALIGNMENT / "alignment.json").write_text(json.dumps(result, indent=2))
    print(f"mapping: film_fps={mp.film_fps:.6f} head={mp.head} break={mp.brk} gap={mp.gap} t0={mp.t0 * 1000:+.2f} ms")
    print(f"frame map flags: {result['frame_map_flags']}")
    return result


def load() -> tuple[FilmMapping, dict]:
    r = json.loads((ALIGNMENT / "alignment.json").read_text())
    return FilmMapping(**r["mapping"]), r
