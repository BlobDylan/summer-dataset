import numpy as np
import pytest

from summer import paradigm
from summer.paradigm import FilmMapping

MP = FilmMapping(film_fps=24000 / 1001, head=25, brk=97210, gap=11020)


def test_filename_convention_is_one_based():
    assert paradigm.filename_to_frame("frame_000919.jpg") == paradigm.ANALYSIS_FIRST == 918
    assert paradigm.frame_to_filename(918) == "frame_000919.jpg"
    assert paradigm.pts(paradigm.ANALYSIS_FIRST) == pytest.approx(36.72)
    assert paradigm.pts(paradigm.ANALYSIS_LAST) == pytest.approx(4763.16)


def test_frame_at_boundaries():
    assert paradigm.frame_at(0.0) == 0
    assert paradigm.frame_at(0.04) == 1          # exact frame onsets must not round down
    assert paradigm.frame_at(36.72) == 918
    assert paradigm.frame_at(0.0399) == 0


def test_film_frames_around_break():
    assert MP.film_frame(0) == 25
    assert MP.film_frame(97209) == 97209 + 25
    assert MP.film_frame(97210) == 97210 + 25 + 11020
    assert MP.film_frame(paradigm.N_FRAMES - 1) == 125742 + 25 + 11020


def test_time_roundtrip_both_segments():
    tp = np.array([0.0, 100.0, 97209.5 * 0.04, 97210.5 * 0.04, 5000.0])
    back = MP.file_time_to_paradigm_time(MP.paradigm_time_to_file_time(tp))
    np.testing.assert_allclose(back, tp, atol=1e-9)


def test_removed_section_has_no_paradigm_time():
    t_gap = (97210 + 25 + 5000) / MP.film_fps     # a film frame inside the removed section
    assert np.isnan(MP.file_time_to_paradigm_time(t_gap))
    assert np.isnan(MP.file_time_to_paradigm_time(0.5))   # before the paradigm's first frame


def test_audio_speed_factor():
    # one paradigm second covers 25/23.976 = 1.0427 seconds of the file
    d = MP.paradigm_time_to_file_time(11.0) - MP.paradigm_time_to_file_time(10.0)
    assert d == pytest.approx(25 * 1001 / 24000)


def test_identity_mapping():
    assert paradigm.IDENTITY.time(918) == pytest.approx(36.72)
    assert paradigm.IDENTITY.film_frame(paradigm.N_FRAMES - 1) == paradigm.N_FRAMES - 1
