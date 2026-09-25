"""The paradigm: the exact version of the movie shown to patients, and its conventions.

Everything in the NWB files that refers to the movie is indexed by *paradigm frame*
``p`` (0-based), where frame ``p`` was on screen from ``pts = p * 0.04`` seconds.
Every derived quantity in this project is stored on that same axis.

Two numbering conventions coexist in the NWB files:
  * arrays (indicator functions) and pts are 0-based:   index p  <->  pts p*0.04
  * frame *filenames* in movie_binning_info are 1-based: "frame_000919.jpg" is p = 918
"""

from dataclasses import dataclass

import numpy as np

FPS = 25
FRAME_S = 1 / FPS
N_FRAMES = 125_743                      # p = 0 .. 125_742
DURATION_S = N_FRAMES * FRAME_S         # 5029.72 s

# analysis window used by the authors (excludes production logos and end credits)
ANALYSIS_FIRST = 918                    # pts 36.72 s  ("frame_000919.jpg")
ANALYSIS_LAST = 119_079                 # pts 4763.16 s ("frame_119080.jpg")


def pts(p):
    return np.asarray(p) * FRAME_S


def frame_at(t):
    """Paradigm frame on screen at paradigm time t (seconds)."""
    return np.floor(np.asarray(t) / FRAME_S + 1e-9).astype(np.int64)


def filename_to_frame(name: str) -> int:
    """'frame_000919.jpg' -> 918."""
    return int(name.rsplit("_", 1)[1].split(".")[0]) - 1


def frame_to_filename(p: int) -> str:
    return f"frame_{p + 1:06d}.jpg"


@dataclass(frozen=True)
class FilmMapping:
    """Paradigm frame -> time in a movie file.

    The paradigm plays the film's frames 1:1 (PAL-style, at 25 fps) with one section
    removed. A copy of the film maps as

        film frame   f(p) = p + head            (p <  brk)
                          = p + head + gap      (p >= brk)
        file time    t(p) = f(p) / film_fps + t0

    ``film_fps`` is the rate at which *film frames* advance in the file's clock: 25 for a
    PAL DVD, 24000/1001 for a film-speed release (even if re-timed to 24 fps). The same
    formula applies to the file's audio, since audio and video share the container clock.
    """

    film_fps: float
    head: int
    brk: int
    gap: int
    t0: float = 0.0

    def film_frame(self, p):
        p = np.asarray(p, dtype=np.int64)
        return p + self.head + np.where(p >= self.brk, self.gap, 0)

    def time(self, p):
        return self.film_frame(p) / self.film_fps + self.t0

    def paradigm_time_to_file_time(self, tp):
        """Continuous version of ``time`` for audio/subtitles (tp in paradigm seconds)."""
        tp = np.asarray(tp, dtype=np.float64)
        pf = tp * FPS
        return (pf + self.head + np.where(pf >= self.brk, self.gap, 0)) / self.film_fps + self.t0

    def file_time_to_paradigm_time(self, t):
        """Inverse of the above; NaN for file times inside the removed section or outside."""
        f = (np.asarray(t, dtype=np.float64) - self.t0) * self.film_fps
        pa = f - self.head
        pb = f - self.head - self.gap
        out = np.where(pa < self.brk, pa, np.where(pb >= self.brk, pb, np.nan))
        out = np.where((out < 0) | (out > N_FRAMES), np.nan, out)
        return out / FPS

    def to_dict(self):
        return dict(film_fps=self.film_fps, head=self.head, brk=self.brk, gap=self.gap, t0=self.t0)


IDENTITY = FilmMapping(film_fps=FPS, head=0, brk=N_FRAMES, gap=0)   # a file that *is* the paradigm movie
