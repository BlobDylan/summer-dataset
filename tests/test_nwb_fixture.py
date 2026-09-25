"""NWB reading and spike-time conversion on a tiny file with the same layout as the real ones."""

import h5py
import numpy as np
import pytest

from summer import neural, nwb, paradigm


@pytest.fixture
def fake_nwb(tmp_path, monkeypatch):
    path = tmp_path / "sub-1.nwb"
    with h5py.File(path, "w") as f:
        g = f.create_group("processing/machine_learning/movie_annotations_indicator_functions")
        on = np.zeros(paradigm.N_FRAMES, np.int64)
        on[[28, 43]] = 1
        run = np.zeros(paradigm.N_FRAMES, np.int64)
        run[100:200] = 1
        g["label_name"] = np.array([b"camera-cuts", b"tom"], dtype=object)
        g["indicator_function"] = np.r_[on, run]
        g["indicator_function_index"] = np.array([paradigm.N_FRAMES, 2 * paradigm.N_FRAMES], np.uint32)
        # watchlog: frames 1..100 every 40 ms, then a 10 s pause at frame 50
        pts = np.arange(1, 101) * 0.04
        neural_ms = np.arange(100) * 40.0
        neural_ms[50:] += 10_000
        w = f.create_group("stimulus/presentation/cleaned_watchlogs")
        w["pts"], w["neural_recording_time"] = pts, neural_ms
        u = f.create_group("units")
        u["unit_id"] = np.array([1, 2])
        for k, v in dict(brain_region=[b"A", b"AH"], hemisphere=[b"L", b"R"], cell_type=[b"x", b"y"]).items():
            u[k] = np.array(v, dtype=object)
        u["is_single_unit"] = np.array([True, False])
        u["spike_times"] = np.array([0.0, 20.0, 1000.0, 5000.0, 12_000.0, 13_000.0, 99_999.0])
        u["spike_times_index"] = np.array([3, 7], np.uint32)
    monkeypatch.setattr(nwb, "nwb_path", lambda p: path)
    monkeypatch.setattr(nwb, "patients", lambda: [1])
    nwb.indicator_functions.cache_clear()
    yield
    nwb.indicator_functions.cache_clear()


def test_indicator_functions_and_intervals(fake_nwb):
    np.testing.assert_array_equal(nwb.onsets("camera-cuts"), [28, 43])
    np.testing.assert_array_equal(nwb.intervals("tom"), [[100, 200]])


def test_spikes_to_paradigm_time(fake_nwb):
    meta, trains = neural.spike_trains(1)
    assert meta["brain_region"] == ["A", "AH"] and list(meta["is_single_unit"]) == [True, False]
    # neural 0 ms -> pts 0.04, 20 ms -> halfway to the next frame, 1000 ms -> 25 frames later
    np.testing.assert_allclose(trains[0], [0.04, 0.06, 1.04])
    # playback pauses after frame 50 (neural 1960 ms) and resumes at 12000 ms (frame 51, pts 2.04):
    # 5000 ms is inside the pause (dropped), 13000 ms is 25 frames after the resume, 99999 is after the end
    np.testing.assert_allclose(trains[1], [2.04, 3.04])
