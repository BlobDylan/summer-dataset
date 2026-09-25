"""Put neural data on the paradigm time axis."""

import numpy as np

from . import nwb

PAUSE_GAP_MS = 200   # consecutive watchlog frames normally ~40 ms apart; longer = playback paused


def neural_to_paradigm_time(patient: int, t_ms: np.ndarray) -> np.ndarray:
    """Neural-recording ms -> paradigm seconds, via the patient's cleaned watchlog.

    Each watchlog row says frame `pts` appeared at neural time `neural_ms`. Between two rows
    time is interpolated linearly. Returns NaN for times before/after the movie and for times
    during a pause (a gap > PAUSE_GAP_MS between consecutive displayed frames), since no movie
    time was advancing then.
    """
    pts, neural = nwb.cleaned_watchlog(patient)
    t_ms = np.asarray(t_ms, dtype=np.float64)
    out = np.interp(t_ms, neural, pts)
    out[(t_ms < neural[0]) | (t_ms > neural[-1])] = np.nan
    j = np.clip(np.searchsorted(neural, t_ms, side="right") - 1, 0, len(neural) - 2)
    paused = (np.diff(neural)[j] > PAUSE_GAP_MS) | (pts[j + 1] == pts[j])
    out[paused] = np.nan
    return out


def spike_trains(patient: int) -> tuple[dict, list[np.ndarray]]:
    """Unit metadata and each unit's spike times in paradigm seconds (paused spikes dropped)."""
    u = nwb.units(patient)
    trains = []
    for st in u.pop("spike_times_ms"):
        t = neural_to_paradigm_time(patient, st)
        trains.append(t[np.isfinite(t)])
    return u, trains
