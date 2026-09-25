"""Export the default viewer tracks: annotations, subtitles, frame-change signal, spikes."""

import json

import numpy as np

from . import movie, neural, nwb, paradigm, tracks
from .paths import ALIGNMENT, PARADIGM, patients

LABEL_GROUPS = {
    "Transitions": ["camera-cuts", "scenes", "days-of-summer"],
    "Characters": ["tom", "summer", "mckenzie", "rachel", "paul", "vance", "autumn", "alison",
                   "douche", "millie", "rhoda", "secretary"],
    "Faces": ["tom-faces", "summer-faces", "mckenzie-faces", "rachel-faces", "paul-faces",
              "vance-faces", "millie-faces"],
    "Speaking": ["tom-speaking", "summer-speaking"],
    "Presence": ["summer-presence", "summer-body-sequence", "persons"],
    "Setting": ["indoor-setting", "the-graduate"],
}


def annotation_track():
    ind = nwb.indicator_functions()
    grouped = {n for g in LABEL_GROUPS.values() for n in g}
    groups = dict(LABEL_GROUPS, Locations=sorted(n for n in ind if n not in grouped))
    labels = {n: [(a * paradigm.FRAME_S, b * paradigm.FRAME_S) for a, b in nwb.intervals(n)] for n in ind}
    tracks.intervals("annotations", "Annotations (NWB)", labels, group="Labels", label_groups=groups,
                     description="Frame-wise labels from the NWB indicator functions")


def subtitle_tracks():
    for lang, name in (("de", "Subtitles DE"), ("en", "Subtitles EN")):
        if not (PARADIGM / f"subs_{lang}.json").exists():
            continue
        segs = json.loads((PARADIGM / f"subs_{lang}.json").read_text())
        tracks.text(f"subs_{lang}", name, [(s["start"], s["end"], s["text"]) for s in segs], group="Text",
                    description="Subtitle file re-timed to paradigm time (not the dub script)")


def frame_change_track():
    video = PARADIGM / "paradigm_movie.mp4"
    if not video.exists():
        return
    _, mad = movie.frame_signal(video, movie.fingerprint(video))
    assert len(mad) == paradigm.N_FRAMES, f"built movie has {len(mad)} frames"
    tracks.series("frame_change", "Frame change (built movie)", t0=0.0, dt=paradigm.FRAME_S, values=mad,
                  group="Stimulus", unit="mean |Δpixel|",
                  description="Difference to the previous frame of the paradigm movie; spikes are cuts. "
                              "Should line up with the camera-cuts label.")


def spike_track():
    pats = []
    for pid in patients():
        meta, trains = neural.spike_trains(pid)
        offs = tracks.write_spike_binary(f"spikes_sub-{pid}.f32", trains)
        units = [dict(unit_id=int(meta["unit_id"][i]), region=meta["brain_region"][i],
                      hemisphere=meta["hemisphere"][i], single_unit=bool(meta["is_single_unit"][i]),
                      cell_type=meta["cell_type"][i], offset=o, count=n)
                 for i, (o, n) in enumerate(offs)]
        pats.append(dict(patient=pid, units=units, bin=f"spikes_sub-{pid}.f32"))
        print(f"  sub-{pid}: {len(units)} units, {sum(n for _, n in offs):,} spikes")
    tracks.spikes("spikes", "Spikes", pats, group="Brain",
                  description="Spike times mapped to paradigm time via each patient's cleaned watchlog; "
                              "spikes during pauses dropped")


def alignment_track():
    fm = np.load(ALIGNMENT / "frame_map.npy")
    brk = json.loads((ALIGNMENT / "alignment.json").read_text())["mapping"]["brk"]

    def runs(mask):
        d = np.diff(np.r_[0, mask.astype(np.int8), 0])
        return [(a * paradigm.FRAME_S, b * paradigm.FRAME_S)
                for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]

    labels = {"maybe ±1 frame": runs(fm["flag"] == 1), "no picture": runs(fm["flag"] == 2),
              "break (7 min removed)": [(brk * paradigm.FRAME_S, (brk + 1) * paradigm.FRAME_S)]}
    tracks.intervals("alignment", "Alignment quality", labels, group="Alignment",
                     label_groups={"Alignment": list(labels)},
                     description="Where the source-to-paradigm mapping is less certain")


def run():
    alignment_track()
    annotation_track()
    subtitle_tracks()
    frame_change_track()
    spike_track()
    print(f"tracks -> {tracks.TRACK_DIR}")
