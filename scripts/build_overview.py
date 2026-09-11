"""Build overview.html: an interactive visual tour of the SUMMER dataset.

Reads every NWB file in data/nwb/, computes summary statistics, and inlines
them into scripts/overview_template.html -> overview.html (repo root).

    uv run --with h5py --with numpy --with scipy python scripts/build_overview.py

Notes on the raw data that this script relies on:
- units/spike_times are in milliseconds on the neural recording clock, which
  keeps running while the movie is paused. cleaned_watchlogs maps that clock
  (neural_recording_time, ms) to movie time (pts, s); spikes that fall inside
  a pause (>1 s gap between displayed frames) are dropped.
- Movie frame k is shown at pts = k * 0.04 s (25 fps, 125,743 frames).
- waveform_mean says "volts" but values are microvolts; the peak is sample 19.
"""

import base64
import glob
import json
import re
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent.parent
NWB_DIR = ROOT / "data" / "nwb"
TEMPLATE = Path(__file__).parent / "overview_template.html"
OUT = ROOT / "overview.html"

FPS = 25
FRAME = 1 / FPS
N_FRAMES = 125_743
MOVIE_S = N_FRAMES * FRAME
T0, T1 = 36.72, 4763.2  # analysis window used by the authors (no credits)
PAUSE_MS = 1000
FS = 32_768

GROUP_OF = {"A": "Amygdala", "AH": "Hippocampus", "MH": "Hippocampus", "PH": "Hippocampus",
            "EC": "Entorhinal cx", "PHC": "Parahippocampal cx",
            "PIC": "Other", "FF": "Other", "LG": "Other", "PRC": "Other"}
GROUPS = ["Amygdala", "Hippocampus", "Entorhinal cx", "Parahippocampal cx", "Other"]
REGION_NAME = {"A": "Amygdala", "AH": "Anterior hippocampus", "MH": "Middle hippocampus",
               "PH": "Posterior hippocampus", "EC": "Entorhinal cortex", "PHC": "Parahippocampal cortex",
               "PIC": "Piriform cortex", "FF": "Fusiform gyrus", "LG": "Lingual gyrus", "PRC": "Perirhinal cortex"}

RASTER_WINDOW = (1980.0, 2160.0)  # 3 minutes of movie time shown in the raster explorer
HEAT_BIN = 30.0
PSTH_BIN = 0.05
PSTH_SPAN = 1.0


def dec(a):
    return [x.decode() if isinstance(x, bytes) else x for x in a]


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def to_movie_time(spikes_ms, nt, pts):
    """Map neural-clock spike times (ms) to movie time (s); NaN during pauses/outside playback."""
    i = np.searchsorted(nt, spikes_ms, side="right") - 1
    ok = (i >= 0) & (i < len(nt) - 1)
    i = np.clip(i, 0, len(nt) - 2)
    gap = nt[i + 1] - nt[i]
    ok &= gap < PAUSE_MS
    frac = (spikes_ms - nt[i]) / np.where(gap > 0, gap, 1)
    t = pts[i] + frac * (pts[i + 1] - pts[i])
    return np.where(ok, t, np.nan)


def label_category(name):
    chars = ["tom", "summer", "mckenzie", "rachel", "paul", "vance", "autumn", "alison",
             "douche", "millie", "rhoda", "secretary"]
    if name.endswith("-faces"):
        return "Faces"
    if name in chars:
        return "Characters"
    if name in ("persons", "summer-presence", "summer-body-sequence"):
        return "Character-related"
    if name.endswith("-speaking"):
        return "Speech"
    if name in ("camera-cuts", "scenes", "days-of-summer"):
        return "Transitions"
    if name == "the-graduate":
        return "Interposed film"
    return "Locations"


def main():
    warnings.filterwarnings("ignore", message="Mean of empty slice")  # neurons silent during the movie
    files =sorted(glob.glob(str(NWB_DIR / "sub-*" / "*.nwb")),
                   key=lambda p: int(re.search(r"sub-(\d+)", p).group(1)))
    if not files:
        sys.exit("No NWB files found; run scripts/download_data.py first.")

    patients, units, bundles, playback = [], [], [], []
    waveforms, movie_spikes = [], []
    ann = None

    for path in files:
        f = h5py.File(path, "r")
        pid = int(f["general/subject/subject_id"][()])
        print(f"sub-{pid}", flush=True)

        if ann is None:
            mi = f["processing/machine_learning/movie_annotations_indicator_functions"]
            names = dec(mi["label_name"][:])
            ind = mi["indicator_function"][:].reshape(len(names), N_FRAMES).astype(np.int16)
            ann = (names, ind)

        cw = f["stimulus/presentation/cleaned_watchlogs"]
        nt, pts = cw["neural_recording_time"][:], cw["pts"][:]
        rw = f["stimulus/presentation/raw_watchlogs"]
        rnt, rpts = rw["neural_recording_time"][:], rw["pts"][:]

        # Pauses (from the cleaned log: consecutive frames >1 s apart on the neural clock)
        dn = np.diff(nt)
        pause_idx = np.where(dn >= PAUSE_MS)[0]
        pauses = [{"wall": r((nt[k] - nt[0]) / 60000, 3), "dur": r(dn[k] / 1000, 1),
                   "movie": r(pts[k] / 60, 3)} for k in pause_idx]
        rdp = np.diff(rpts)
        n_back = int((rdp < -0.5).sum())
        n_fwd = int((rdp > 0.5).sum())

        # Downsampled playback path from the raw log (wall-clock vs movie position)
        rdn = np.diff(rnt)
        keep = np.zeros(len(rpts), bool)
        keep[::125] = True
        jump = np.where((np.abs(rdp - FRAME) > 0.1) | (rdn > 300))[0]
        keep[jump] = True
        keep[np.minimum(jump + 1, len(rpts) - 1)] = True
        keep[[0, -1]] = True
        playback.append({"id": pid,
                         "wall": np.round((rnt[keep] - rnt[0]) / 60000, 3).tolist(),
                         "movie": np.round(rpts[keep] / 60, 3).tolist()})

        # Electrodes -> bundles
        e = f["general/extracellular_ephys/electrodes"]
        e_region, e_group, e_hemi = dec(e["brain_region"][:]), dec(e["group_name"][:]), dec(e["hemisphere"][:])
        e_csc = e["csc_nr"][:]
        ex, ey, ez = e["x"][:], e["y"][:], e["z"][:]

        u = f["units"]
        u_region = dec(u["brain_region"][:])
        u_csc = u["csc_nr"][:]
        u_su = u["is_single_unit"][:]
        u_ct = dec(u["cell_type"][:])
        u_hemi = dec(u["hemisphere"][:])
        snr, isi, cv2, iso = u["peak_SNR"][:], u["isi_violations"][:], u["cv2"][:], u["iso_dist"][:]
        wf = u["waveform_mean"][:]
        idx = u["spike_times_index"][:]
        st_all = u["spike_times"][:]
        starts = np.concatenate([[0], idx[:-1]])

        csc_to_bundle = {c: g for c, g in zip(e_csc, e_group)}
        for g in dict.fromkeys(e_group):
            m = np.array([x == g for x in e_group])
            unit_m = np.isin(u_csc, e_csc[m])
            reg = e_region[int(np.where(m)[0][0])]
            bundles.append({"p": pid, "name": g, "region": reg, "group": GROUP_OF[reg],
                            "hemi": e_hemi[int(np.where(m)[0][0])],
                            "x": r(ex[m].mean(), 1), "y": r(ey[m].mean(), 1), "z": r(ez[m].mean(), 1),
                            "units": int(unit_m.sum()), "wires": int(m.sum())})

        n_spk_total = 0
        for k in range(len(u_region)):
            s = st_all[starts[k]:idx[k]]
            n_spk_total += len(s)
            t = to_movie_time(s, nt, pts)
            t = np.sort(t[np.isfinite(t) & (t >= T0) & (t < T1)])
            movie_spikes.append(t)
            w = wf[k].astype(float)
            peak = w[19]
            y = np.sign(peak) * w
            width = (np.argmin(y[19:]) / FS) * 1000
            waveforms.append(np.round(127 * w / (np.abs(peak) + 1e-9)).clip(-127, 127).astype(np.int8))
            reg = u_region[k]
            units.append({"p": pid, "region": reg, "group": GROUP_OF[reg], "hemi": u_hemi[k],
                          "su": bool(u_su[k]), "type": u_ct[k], "csc": int(u_csc[k]),
                          "bundle": csc_to_bundle.get(u_csc[k]),
                          "rate": r(len(t) / (T1 - T0), 4), "snr": r(snr[k], 2), "isi": r(isi[k], 3),
                          "cv2": r(cv2[k], 3), "iso": r(iso[k], 2), "width": r(width, 3),
                          "amp": r(abs(peak), 1)})

        age = int(re.match(r"P(\d+)Y", f["general/subject/age"][()].decode()).group(1))
        pu = [x for x in units if x["p"] == pid]
        patients.append({
            "id": pid, "age": age, "sex": f["general/subject/sex"][()].decode(),
            "year": int(f["session_start_time"][()].decode()[:4]),
            "system": "Cheetah" if pid <= 16 else "Pegasus",
            "units": len(pu), "su": sum(x["su"] for x in pu), "bundles": len(set(e_group)),
            "by_group": {g: sum(x["group"] == g for x in pu) for g in GROUPS},
            "regions": sorted(set(u_region), key=list(REGION_NAME).index),
            "spikes": int(n_spk_total),
            "session_min": r((nt[-1] - nt[0]) / 60000, 2),
            "pauses": pauses, "pause_total_s": r(dn[pause_idx].sum() / 1000, 1),
            "skips_back": n_back, "skips_fwd": n_fwd,
            "mean_rate": r(np.mean([x["rate"] for x in pu]), 3),
        })
        f.close()

    N = len(units)
    unit_p = np.array([x["p"] for x in units])
    unit_g = np.array([GROUPS.index(x["group"]) for x in units])
    rates = np.array([x["rate"] for x in units])
    print(f"{N} units", flush=True)

    # ---------------- Annotations ----------------
    names, ind = ann
    present = (ind == 1)
    labels = []
    for i, n in enumerate(names):
        on = np.flatnonzero(np.diff(present[i].astype(np.int8)) == 1) + 1
        if present[i, 0]:
            on = np.r_[0, on]
        labels.append({"name": n, "cat": label_category(n), "pct": r(100 * present[i].mean(), 3),
                       "onsets": int(len(on)), "seconds": r(present[i].sum() * FRAME, 1)})
    bar_bin = 25  # 1-second columns
    n_cols = N_FRAMES // bar_bin
    barcode = present[:, :n_cols * bar_bin].reshape(len(names), n_cols, bar_bin).mean(2)
    barcode_u8 = np.round(barcode * 255).astype(np.uint8)

    cut_t = np.flatnonzero(present[names.index("camera-cuts")]) * FRAME
    scene_t = np.flatnonzero(present[names.index("scenes")]) * FRAME
    shots = np.diff(cut_t)
    scenes_len = np.diff(scene_t)

    co_names = ["tom", "summer", "persons", "summer-presence", "tom-faces", "summer-faces", "tom-speaking",
                "summer-speaking", "mckenzie", "indoor-setting", "office", "tom-apartment", "park",
                "karaoke-bar", "ikea", "days-of-summer"]
    X = np.stack([present[names.index(n)].astype(float) for n in co_names])
    phi = np.corrcoef(X)

    # ---------------- Binned spike counts ----------------
    def bin_counts(width):
        edges = np.arange(T0, T1 + 1e-9, width)
        return np.stack([np.histogram(s, edges)[0] for s in movie_spikes]).astype(np.float32), edges

    C1, _ = bin_counts(1.0)                                     # N x 4726 (1 s)
    nb = C1.shape[1] // int(HEAT_BIN)
    C10 = C1[:, :nb * int(HEAT_BIN)].reshape(N, nb, int(HEAT_BIN)).sum(2)
    z10 = (C10 - C10.mean(1, keepdims=True)) / (C10.std(1, keepdims=True) + 1e-6)
    heat_order = np.lexsort((-rates, unit_p, unit_g))
    heat_i8 = np.round(z10[heat_order] * 40).clip(-127, 127).astype(np.int8)

    # Slow drift: first vs last fifth of the movie, and share of 10-s count variance that is slow (>5 min)
    active = np.flatnonzero(rates >= 0.5)
    fifth = C1.shape[1] // 5
    lr_drift = np.log2((C1[active, -fifth:].mean(1) + 0.05) / (C1[active, :fifth].mean(1) + 0.05))
    C10s = C1[active, :(C1.shape[1] // 10) * 10].reshape(len(active), -1, 10).sum(2)
    k = 31
    slow = np.stack([np.convolve(np.pad(c, (k // 2, k // 2), mode="edge"), np.ones(k) / k, "valid") for c in C10s])
    drift = {"n": int(len(active)), "up15": r((lr_drift > np.log2(1.5)).mean(), 3),
             "down15": r((lr_drift < -np.log2(1.5)).mean(), 3), "x2": r((np.abs(lr_drift) > 1).mean(), 3),
             "slow_var": r(np.median(slow.var(1) / C10s.var(1)), 3)}

    # Sparsity at the paper's bin lengths
    sparsity = []
    for w in [0.04, 0.08, 0.2, 0.48, 1.0]:
        n_bins = int((T1 - T0) // w)
        empty = np.array([1 - len(np.unique(((s - T0) // w).astype(int))) / n_bins for s in movie_spikes])
        mean = np.array([len(s) / n_bins for s in movie_spikes])
        q = lambda a: [r(v, 4) for v in np.percentile(a, [10, 25, 50, 75, 90])]
        sparsity.append({"ms": int(round(w * 1000)), "bins": n_bins, "empty": q(empty), "mean": q(mean)})

    # Autocorrelation: labels and neural activity (lags 0..180 s, 1 s bins)
    max_lag = 180
    lags = np.arange(0, max_lag + 1)

    def acf(x):
        x = x - x.mean()
        d = (x * x).sum()
        return np.array([1.0 if L == 0 else (x[:-L] * x[L:]).sum() / d for L in lags])

    frames_1s = present[:, int(T0 / FRAME):int(T1 / FRAME)]
    frames_1s = frames_1s[:, :(frames_1s.shape[1] // 25) * 25].reshape(len(names), -1, 25).mean(2)
    acf_labels = {n: [r(v, 4) for v in acf(frames_1s[names.index(n)])]
                  for n in ["summer", "tom", "persons", "indoor-setting", "tom-speaking", "office"]}
    Z1 = (C1 - C1.mean(1, keepdims=True)) / (C1.std(1, keepdims=True) + 1e-6)
    good = rates >= 0.1
    acf_units = np.stack([acf(Z1[i]) for i in np.flatnonzero(good)])
    # population-vector similarity: per patient, correlate z-scored population vectors at t and t+lag
    pv = []
    for p in sorted(set(unit_p)):
        Zp = Z1[(unit_p == p) & good]
        Zp = Zp - Zp.mean(0, keepdims=True)
        nrm = np.linalg.norm(Zp, axis=0) + 1e-9
        pv.append([1.0] + [float(np.mean((Zp[:, :-L] * Zp[:, L:]).sum(0) / (nrm[:-L] * nrm[L:]))) for L in lags[1:]])
    acf_neural = {"unit_median": [r(v, 4) for v in np.median(acf_units, 0)],
                  "unit_q25": [r(v, 4) for v in np.percentile(acf_units, 25, 0)],
                  "unit_q75": [r(v, 4) for v in np.percentile(acf_units, 75, 0)],
                  "popvec": [r(v, 4) for v in np.mean(pv, 0)]}

    # Shared movie-locked activity: each patient's population rate vs the mean of all other patients
    def highpass(x, win=60):
        k = np.ones(win) / win
        pad = np.pad(x, (win // 2, win - win // 2 - 1), mode="edge")
        return x - np.convolve(pad, k, mode="valid")

    pids = sorted(set(unit_p))
    pop = np.stack([highpass(Z1[(unit_p == p) & good].mean(0)) for p in pids])
    rng = np.random.default_rng(0)
    T = pop.shape[1]
    isc = []
    for i, p in enumerate(pids):
        others = np.delete(pop, i, 0).mean(0)
        null = [np.corrcoef(np.roll(pop[i], rng.integers(600, T - 600)), others)[0, 1] for _ in range(200)]
        isc.append({"p": int(p), "r": r(np.corrcoef(pop[i], others)[0, 1], 4),
                    "lo": r(np.percentile(null, 2.5), 4), "hi": r(np.percentile(null, 97.5), 4),
                    "n": int(((unit_p == p) & good).sum())})

    # ---------------- Event-aligned responses ----------------
    def onsets(name, min_absent=1.0):
        x = present[names.index(name)]
        on = np.flatnonzero(np.diff(x.astype(np.int8)) == 1) + 1
        need = int(round(min_absent / FRAME))
        keep = [o for o in on if o >= need and not x[o - need:o].any()]
        return np.array(keep) * FRAME

    def switches(name):
        x = ind[names.index(name)]
        valid = x != 99
        idx_ = np.flatnonzero(valid[1:] & valid[:-1] & (np.diff(x) != 0)) + 1
        return idx_ * FRAME

    events = {
        "Camera cut": onsets("camera-cuts"),
        "Scene cut": onsets("scenes"),
        "Summer appears": onsets("summer"),
        "Tom appears": onsets("tom"),
        "Summer starts speaking": onsets("summer-speaking"),
        "Tom starts speaking": onsets("tom-speaking"),
        "Day-counter card": onsets("days-of-summer"),
        "Indoor/outdoor switch": switches("indoor-setting"),
    }
    edges_rel = np.arange(-PSTH_SPAN, PSTH_SPAN + 1e-9, PSTH_BIN)
    nbins = len(edges_rel) - 1
    psth, responsive, cut_heat = {}, {}, None
    for ename, ev in events.items():
        ev = ev[(ev >= T0 + PSTH_SPAN) & (ev < T1 - PSTH_SPAN)]
        E = (ev[:, None] + edges_rel[None, :]).ravel()
        ratio = np.full((N, nbins), np.nan, np.float32)
        pvals = np.ones(N)
        for i, s in enumerate(movie_spikes):
            c = np.diff(np.searchsorted(s, E).reshape(len(ev), -1), axis=1)  # events x bins
            if rates[i] > 0:
                ratio[i] = c.mean(0) / PSTH_BIN / rates[i]
            pre, post = c[:, :nbins // 2].sum(1), c[:, nbins // 2:].sum(1)
            if rates[i] >= 0.1 and np.any(pre != post):
                pvals[i] = wilcoxon(post, pre, zero_method="zsplit").pvalue
        sig = pvals < 0.01
        direction = np.nanmean(ratio[:, nbins // 2:], 1) > np.nanmean(ratio[:, :nbins // 2], 1)
        out = {"n_events": int(len(ev)), "groups": {}}
        resp = {}
        for gi, g in enumerate(GROUPS):
            m = (unit_g == gi) & (rates >= 0.1)
            R = ratio[m]
            out["groups"][g] = {"mean": [r(v, 4) for v in np.nanmean(R, 0)],
                                "sem": [r(v, 4) for v in np.nanstd(R, 0) / np.sqrt(m.sum())]}
            resp[g] = {"n": int(m.sum()), "sig": int(sig[m].sum()), "up": int((sig & direction)[m].sum())}
        m = rates >= 0.1
        resp["All"] = {"n": int(m.sum()), "sig": int(sig[m].sum()), "up": int((sig & direction)[m].sum())}
        psth[ename] = out
        responsive[ename] = resp
        if ename == "Camera cut":
            # relative to each neuron's own pre-event second, sorted by the 0-500 ms response
            rel = ratio / np.nanmean(ratio[:, :nbins // 2], 1, keepdims=True)
            rel = np.where(np.isfinite(rel) & (rates[:, None] >= 0.1), rel, 1.0)  # too few spikes -> neutral
            post_mean = rel[:, nbins // 2:nbins // 2 + 10].mean(1)
            order = np.lexsort((-post_mean, unit_g))
            lr = np.log2(np.clip(rel, 0.25, 4))
            cut_heat = {"data": b64(np.round(lr[order] * 63).clip(-127, 127).astype(np.int8)),
                        "order": order.tolist(), "sig": sig[order].astype(int).tolist()}
        print(ename, len(ev), resp["All"], flush=True)

    # ---------------- Raster explorer window ----------------
    w0, w1 = RASTER_WINDOW
    raster = {"t0": w0, "t1": w1, "patients": {}}
    for p in pids:
        ids = np.flatnonzero(unit_p == p)
        ids = ids[np.lexsort((-rates[ids], unit_g[ids]))]
        chunks, offsets = [], [0]
        for i in ids:
            s = movie_spikes[i]
            s = s[(s >= w0) & (s < w1)]
            chunks.append(np.round((s - w0) * 200).astype(np.uint16))  # 5 ms resolution
            offsets.append(offsets[-1] + len(s))
        raster["patients"][str(p)] = {"units": ids.tolist(), "offsets": offsets,
                                      "spikes": b64(np.concatenate(chunks) if chunks else np.zeros(0, np.uint16))}
    f0, f1 = int(w0 / FRAME), int(w1 / FRAME)
    raster_tracks = {}
    for n in ["summer", "tom", "summer-faces", "tom-faces", "summer-speaking", "tom-speaking", "persons",
              "indoor-setting", "camera-cuts", "scenes"]:
        x = present[names.index(n), f0:f1]
        if n in ("camera-cuts", "scenes"):
            raster_tracks[n] = {"points": [r(v, 2) for v in np.flatnonzero(x) * FRAME]}
        else:
            d = np.diff(np.r_[0, x.astype(np.int8), 0])
            raster_tracks[n] = {"spans": [[r(a * FRAME, 2), r(b * FRAME, 2)]
                                          for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]}

    # neurons per wire
    wire_counts = {}
    for x in units:
        wire_counts[(x["p"], x["csc"])] = wire_counts.get((x["p"], x["csc"]), 0) + 1
    per_wire = np.bincount(list(wire_counts.values()))

    data = {
        "meta": {"n_units": N, "n_su": int(sum(x["su"] for x in units)), "n_patients": len(patients),
                 "n_bundles": len(bundles), "n_wires": len(wire_counts),
                 "spikes_total": int(sum(p["spikes"] for p in patients)),
                 "movie_s": MOVIE_S, "t0": T0, "t1": T1, "n_frames": N_FRAMES,
                 "groups": GROUPS, "region_names": REGION_NAME,
                 "pauses_total": int(sum(len(p["pauses"]) for p in patients))},
        "patients": patients, "units": units, "bundles": bundles, "playback": playback,
        "waveforms": b64(np.stack(waveforms)),
        "per_wire": per_wire.tolist(),
        "labels": labels, "barcode": {"cols": int(n_cols), "data": b64(barcode_u8)},
        "shots": [r(v, 2) for v in shots], "scene_lengths": [r(v, 2) for v in scenes_len],
        "cooc": {"names": co_names, "phi": [[r(v, 3) for v in row] for row in phi]},
        "sparsity": sparsity,
        "acf": {"lags": lags.tolist(), "labels": acf_labels, "neural": acf_neural},
        "heat": {"bin": HEAT_BIN, "cols": int(nb), "order": heat_order.tolist(), "data": b64(heat_i8)},
        "drift": drift,
        "isc": isc,
        "psth": {"edges": [r(v, 3) for v in edges_rel], "events": psth, "responsive": responsive},
        "cut_heat": cut_heat,
        "raster": raster, "raster_tracks": raster_tracks,
    }
    payload = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload)
    OUT.write_text('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + html)
    print(f"Wrote {OUT} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
