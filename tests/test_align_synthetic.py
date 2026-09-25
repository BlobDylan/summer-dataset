"""End-to-end alignment on synthetic movie copies with known mappings (no data files needed).

A fake film has shot changes at known frames; the paradigm is that film with a head offset
and a removed section. Each "copy" presents the film frames at some rate (PAL, film speed,
or re-timed to 24 fps with duplicated frames). fit() must recover the exact mapping.
"""

import numpy as np
import pytest

from summer import align, nwb, paradigm, verify
from summer.paradigm import FilmMapping

N = paradigm.N_FRAMES


def synthetic(mp: FilmMapping, kind: str, seed=0):
    rng = np.random.default_rng(seed)
    # paradigm annotations
    cuts = np.cumsum(rng.integers(40, 300, 2000))
    cuts = cuts[(cuts > 30) & (cuts < N - 100) & (np.abs(cuts - mp.brk) > 60)]
    ind = {k: np.zeros(N, np.uint8) for k in ("camera-cuts", "scenes", "days-of-summer")}
    ind["camera-cuts"][cuts] = 1
    ind["scenes"][mp.brk] = 1                       # the removed section ends at a scene change
    ind["days-of-summer"][mp.brk: mp.brk + 50] = 1
    # film: shot changes at the mapped cuts, the break's landing frame, and inside the removed part
    n_film = int(mp.film_frame(N - 1)) + 40
    film_cut = np.zeros(n_film, bool)
    film_cut[mp.film_frame(cuts)] = True
    film_cut[mp.film_frame(mp.brk)] = True
    removed = np.arange(mp.brk + mp.head, mp.brk + mp.head + mp.gap)
    film_cut[rng.choice(removed, mp.gap // 150, replace=False)] = True
    film_cut[rng.choice(mp.head, 3, replace=False) if mp.head > 3 else []] = True
    # the copy: which film frame each file frame shows, and when
    if kind == "retimed24":                         # 23.976 film re-timed to 24 fps (duplicates)
        n = np.arange(int(n_film * 1.001) - 2)
        shown, pts = np.round(n * 1000 / 1001).astype(int), n / 24
    else:
        shown = np.arange(n_film)
        pts = shown / (25.0 if kind == "pal" else 24000 / 1001)
    shown = np.clip(shown, 0, n_film - 1)
    new = np.r_[True, np.diff(shown) != 0]
    mad = np.where(new, rng.uniform(0.5, 3.0, len(shown)), 0.0).astype(np.float32)
    mad[new & film_cut[shown]] = rng.uniform(20, 60, (new & film_cut[shown]).sum())
    return pts, mad, ind


CASES = [
    ("retimed24", FilmMapping(24000 / 1001, 25, 97210, 11020)),     # the copy used in this repo
    ("pal", FilmMapping(25.0, 0, 97212, 11020)),                    # the Cine Project DVD layout
    ("native", FilmMapping(24000 / 1001, 140, 60000, 5000)),        # a different, made-up edit
]


@pytest.mark.parametrize("kind,true", CASES, ids=[c[0] for c in CASES])
def test_fit_recovers_mapping(monkeypatch, kind, true):
    pts, mad, ind = synthetic(true, kind)
    monkeypatch.setattr(nwb, "indicator_functions", lambda: ind)
    mp, diag = align.fit(pts, mad)
    assert mp.film_fps == pytest.approx(true.film_fps)
    assert (mp.head, mp.brk, mp.gap) == (true.head, true.brk, true.gap)
    assert abs(mp.t0) < 0.5 / mp.film_fps

    res = verify.check_cuts(mp, pts, mad)
    assert res["pass"]
    assert res["control_no_gap"]["B"]["exact"] < 0.2


def test_verify_rejects_wrong_mapping(monkeypatch):
    kind, true = CASES[0]
    pts, mad, ind = synthetic(true, kind)
    monkeypatch.setattr(nwb, "indicator_functions", lambda: ind)
    wrong = FilmMapping(true.film_fps, true.head + 2, true.brk, true.gap)
    assert not verify.check_cuts(wrong, pts, mad)["pass"]
