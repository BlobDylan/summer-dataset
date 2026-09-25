"""Write data tracks for the viewer. All times are paradigm seconds (paradigm frame p = p * 0.04 s).

A track is one JSON file in data/derived/viewer/tracks/ (plus an optional binary file).
The viewer lists every file in that folder, so adding a track is just calling one of these:

    from summer import tracks
    tracks.text("vlm_captions", "VLM scene captions", segments=[(start_s, end_s, "text"), ...])
    tracks.series("clip_novelty", "CLIP novelty", t0=0.0, dt=0.08, values=array, group="Stimulus")
    tracks.intervals("my_labels", "My labels", {"label": [(start_s, end_s), ...]}, group="Labels")

Types the viewer can draw: intervals, text, series, spikes. A new *type* needs a renderer
in src/summer/viewer_app/app.js (see `RENDERERS` there).
"""

import json

import numpy as np

from .paths import VIEWER_DATA

TRACK_DIR = VIEWER_DATA / "tracks"


def _write(track: dict) -> dict:
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    (TRACK_DIR / f"{track['id']}.json").write_text(json.dumps(track, separators=(",", ":")))
    return track


def intervals(id: str, name: str, labels: dict[str, list[tuple[float, float]]], group: str = "Labels",
              label_groups: dict[str, list[str]] | None = None, description: str = "") -> dict:
    """Named on/off labels. `label_groups` optionally groups label names for filtering."""
    return _write(dict(id=id, name=name, type="intervals", group=group, description=description,
                       label_groups=label_groups or {name: sorted(labels)},
                       labels={k: [[round(float(a), 3), round(float(b), 3)] for a, b in v]
                               for k, v in labels.items()}))


def text(id: str, name: str, segments: list[tuple[float, float, str]], group: str = "Text",
         description: str = "") -> dict:
    """Timed text (subtitles, captions, VLM descriptions)."""
    segs = sorted((round(float(a), 3), round(float(b), 3), str(t)) for a, b, t in segments)
    return _write(dict(id=id, name=name, type="text", group=group, description=description,
                       segments=[list(s) for s in segs]))


def series(id: str, name: str, t0: float, dt: float, values, group: str = "Series", unit: str = "",
           description: str = "") -> dict:
    """A regularly sampled numeric signal, sample i at t0 + i*dt. NaN = missing."""
    v = np.asarray(values, dtype=np.float32)
    return _write(dict(id=id, name=name, type="series", group=group, unit=unit, description=description,
                       t0=float(t0), dt=float(dt),
                       values=[None if not np.isfinite(x) else round(float(x), 5) for x in v]))


def spikes(id: str, name: str, patients: list[dict], group: str = "Brain", description: str = "") -> dict:
    """Spike trains. Each patient: {"patient": int, "units": [...unit meta with offset/count...],
    "bin": "<file>.f32"} where the .f32 file (in the tracks folder) holds every unit's sorted
    spike times back to back, as little-endian float32 paradigm seconds."""
    return _write(dict(id=id, name=name, type="spikes", group=group, description=description,
                       patients=patients))


def write_spike_binary(filename: str, trains: list[np.ndarray]) -> list[tuple[int, int]]:
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    offsets, pos = [], 0
    with open(TRACK_DIR / filename, "wb") as f:
        for t in trains:
            a = np.sort(np.asarray(t, dtype="<f4"))
            f.write(a.tobytes())
            offsets.append((pos, len(a)))
            pos += len(a)
    return offsets
