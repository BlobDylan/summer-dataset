# Aligning a movie copy to the paradigm

## The paradigm and its conventions

The **paradigm** is the exact version of the movie shown to the patients: German dub, 25 fps,
125,743 frames, 5029.72 s. The NWB files never refer to a video file; everything is indexed by the
**paradigm frame `p`**, 0-based, on screen from `p × 0.04 s`:

- the 53 annotation indicator functions (`processing/machine_learning/movie_annotations_indicator_functions`),
- the bin edges (`movie_binning_info`), and
- each patient's watchlogs (`stimulus/presentation/cleaned_watchlogs`), which link `pts` to neural time.

Watch out for the two numbering conventions: arrays and `pts` are 0-based, but the frame *filenames*
in `movie_binning_info` are 1-based (`frame_000919.jpg` is `p = 918`). The authors' analysis window
is `p = 918 … 119,079` (36.72 – 4763.16 s). `summer.paradigm` holds these constants and helpers.

## The mapping

The paradigm plays the film's frames one-to-one at 25 fps (PAL speed-up: 4.27 % faster than the
film's 23.976 fps), starting some frames in, with one section removed. For a movie copy:

```
film frame  f(p) = p + head            (p <  brk)
                 = p + head + gap      (p >= brk)
time in the copy   t(p) = f(p) / film_fps + t0       (audio and video share this clock)
```

`film_fps` is the rate at which *film frames* advance in the copy: 25 for a PAL DVD, 23.976 for a
film-speed release — including one re-timed to 24 fps, which only duplicates a frame every ~1000.
`summer.paradigm.FilmMapping` implements this; `summer run align` measures it.

For the copy in the shared Drive (`known_copies.json`):

| | |
|---|---|
| film_fps | 23.976 (the file is a 24 fps re-time with duplicate/drop bursts every ~1000 frames) |
| head | 25 film frames (≈ 1.04 s) |
| brk | paradigm frame 97,210 |
| gap | 11,020 film frames (7 min 20 s the patients never saw) |
| t0 | −0.4 ms |

The published [`wrapper.py`](https://github.com/mormannlab/SUMMER/blob/main/movie_wrapper/wrapper.py)
encodes the same layout for the German Cine Project DVD, but with the break at 97,212. The annotations
put a scene change and a Days-of-Summer title card at 97,210, and the film has a hard cut into that
card exactly where 97,210 lands, so this repo uses 97,210 (it only changes what frames 97,210–97,211 show).
Don't use the published `remap_video_frames.py`: it applies the paradigm→DVD mapping to DVD frames
(shifting them the wrong way), overwrites frames while renaming in place, and assumes a 25 fps source.

## How it is measured

1. Decode the copy once at 96×40 greyscale; `mad[i]` = mean absolute difference to the previous frame.
   Hard cuts are spikes (a detector keeps spikes that dominate both neighbours).
2. For each candidate `film_fps` (25, 23.976, 24), histogram `(film position of each detected cut) −
   (each annotated camera cut)`. The right rate gives sharp integer peaks, one per contiguous segment:
   `head`, and `head + gap` after the removed section.
3. `t0` is the median sub-frame residual of the matched cuts.
4. The break lies between the last cut matched in the first segment and the first in the second; the
   frame where annotated transitions (cuts, scene changes, title cards) best land on transitions in the
   copy wins.

## How it is verified (`report.html`)

`summer setup` runs these on the copy, and again on the built paradigm movie with the identity mapping
(round trip). It stops if they fail.

- **Cuts**: for every annotated cut, where is the largest frame difference within ±5 frames of the
  predicted frame? Pass: ≥ 80 % exactly on it, in both segments. Controls must fail: the same test with
  the head shifted by ±1 frame (≤ 10 %), without the gap (second segment collapses), at random frames.
- **Audio**: 10 ms voice activity (WebRTC VAD) on the German track vs the annotated `tom-speaking` /
  `summer-speaking`. Pass: correlation well above the same test at the wrong speed (off by the PAL factor),
  and annotated speech *offsets* within 0.15 s of detected ones. (Annotated *onsets* lead detected speech
  by ~0.3 s in both languages while offsets agree, so that is how the labels were drawn, not an audio shift.)
- **Contact sheets**: all 34 Days-of-Summer title cards, the frames around the break, random frames with
  their annotated faces, and pairs of frames around annotated cuts. Look at them once.

Results for the Drive copy: 87.7 % / 93.9 % of cuts exact (segment A / B), controls 0.6–2.7 %, no-gap
control 8.9 % in segment B, 32/34 title-card onsets exact; built movie 88.9 %; speech offsets within
0.07 s; the built audio matches the mapped source sample-exactly (cross-correlation lag 0.00 ms).

## Known limits

- **Re-timed copies**: around each duplicate/drop burst a frame can be off by one (40 ms). These frames
  are flagged (`frame_map.npy` flag 1, "maybe ±1 frame" in the viewer); 1.3 % of the Drive copy.
  Where no motion reveals a burst the neighbouring frames look the same anyway.
- The last 15 paradigm frames (0.6 s of end credits) are past the end of the Drive copy (flag 2, black).
- The remaining ~10 % of cuts that don't hit exactly are fades, split-screen montage panels, and cuts
  hidden by foreground motion — not misalignment (see the contact sheets).
- **Audio pitch**: the audio is sped up by resampling, like PAL playback, so pitch is 4.27 % higher than
  the film. Whether the original presentation used pitch correction is unknown; timing is identical.
- **Playback speed during the experiment**: the watchlogs show ~41 ms per frame in real time, not 40.
  Spike times are mapped through each patient's watchlog, so this is accounted for.
