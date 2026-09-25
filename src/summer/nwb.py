"""Read the pieces of the SUMMER NWB files we need, with plain h5py (fast, no pynwb)."""

from functools import lru_cache

import h5py
import numpy as np

from . import paradigm
from .paths import nwb_path, patients


def _str(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def _ragged(group, name):
    data = group[name][:]
    ends = group[f"{name}_index"][:]
    starts = np.r_[0, ends[:-1]]
    return [data[s:e] for s, e in zip(starts, ends)]


@lru_cache
def indicator_functions() -> dict[str, np.ndarray]:
    """label -> uint8 array of length N_FRAMES, indexed by paradigm frame.

    Identical in every patient's file, so read from the first one.
    """
    with h5py.File(nwb_path(patients()[0]), "r") as f:
        g = f["processing/machine_learning/movie_annotations_indicator_functions"]
        names = _str(g["label_name"][:])
        funcs = _ragged(g, "indicator_function")
    out = {n: v.astype(np.uint8) for n, v in zip(names, funcs)}
    assert all(len(v) == paradigm.N_FRAMES for v in out.values())
    return out


def onsets(label: str) -> np.ndarray:
    """Paradigm frames where a label switches on (for cuts: the first frame of the new shot)."""
    x = indicator_functions()[label].astype(np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x]) == 1)


def intervals(label: str) -> np.ndarray:
    """(n, 2) array of [first, last+1) paradigm frames where the label is on."""
    x = indicator_functions()[label].astype(np.int8)
    d = np.diff(np.r_[0, x, 0])
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1)


def units(patient: int) -> dict:
    """Unit metadata and spike times (neural-recording ms, 0 = movie start) for one patient."""
    with h5py.File(nwb_path(patient), "r") as f:
        u = f["units"]
        return dict(
            unit_id=u["unit_id"][:],
            brain_region=_str(u["brain_region"][:]),
            hemisphere=_str(u["hemisphere"][:]),
            is_single_unit=u["is_single_unit"][:].astype(bool),
            cell_type=_str(u["cell_type"][:]),
            spike_times_ms=_ragged(u, "spike_times"),
        )


def cleaned_watchlog(patient: int) -> tuple[np.ndarray, np.ndarray]:
    """(pts seconds, neural ms) for every displayed frame, pauses collapsed (pts repeats)."""
    with h5py.File(nwb_path(patient), "r") as f:
        g = f["stimulus/presentation/cleaned_watchlogs"]
        return g["pts"][:], g["neural_recording_time"][:]
