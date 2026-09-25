# AGENTS.md: using the SUMMER data in this repo

You were probably sent here to run an experiment on **SUMMER**: single-neuron recordings from 29
epilepsy patients (2,286 units, medial temporal lobe) made while each watched the whole movie
*(500) Days of Summer* (DANDI 001616, Scientific Data 2026).

The most important thing this repo provides is **the movie frames aligned to the neural data**: a
rebuilt copy of the exact movie version the patients saw, where video frame `p` *is* paradigm frame
`p`, the index every annotation and spike in the dataset refers to. Use it for anything that
needs the stimulus (images, audio, text, features). You never need to align anything yourself.

Run everything from the repo root with `uv run` (e.g. `uv run python my_script.py`). The package is
`summer` (`src/summer/`).

## 0. Check the data is ready

```bash
uv run summer status
```

- Every step `ok`: go ahead.
- A step `missing`/`stale`: run `uv run summer setup` (about 25 min from scratch, seconds if little changed).
- An input missing (NWB or movie): NWB can be fetched with `uv run summer download-nwb`. The **movie cannot
  be downloaded** (it is copyrighted); ask the user to put it in `data/movie/` (see `docs/DATA.md`). Do not
  look for it online.

## 1. The time axis: paradigm frame `p`

The **paradigm** is the movie version shown to the patients: 25 fps, **125,743 frames** (`p = 0 … 125,742`),
5029.72 s, German dub. Frame `p` is on screen from **`t = p × 0.04 s`** (0-based). All of the following
use this axis:

- annotation arrays: element `p`;
- spike times (after `neural.spike_trains`): seconds on this clock;
- bin edges in `movie_binning_info`: seconds on this clock;
- the rebuilt movie, audio and subtitles.

`summer.paradigm` has the constants: `FPS`, `FRAME_S` (0.04), `N_FRAMES`, `ANALYSIS_FIRST` (918),
`ANALYSIS_LAST` (119,079), `pts(p)`, `frame_at(t)`, `filename_to_frame`.

- **The analysis window**: the authors analyse `p = 918 … 119,079` (36.72 – 4763.16 s), which leaves out the
  logos and end credits. Use it unless you have a reason not to.
- **Off-by-one trap**: `movie_binning_info` names frames by 1-based *filenames*: `"frame_000919.jpg"` is
  `p = 918`. Convert with `paradigm.filename_to_frame`. Arrays and times are 0-based.

## 2. The aligned frames

`data/derived/paradigm/paradigm_movie.mp4`: H.264, 1728×720 (2.4:1, no black bars), 25 fps, exactly
125,743 frames, a keyframe every 25 frames. **The i-th decoded frame is paradigm frame i**, and its
timestamp is `i × 0.04`. It has two audio tracks: German (default; what the patients heard) and English.

Read frames with PyAV (installed; it bundles FFmpeg, and there is no system `ffmpeg`):

```python
import av, numpy as np
from summer import paradigm
MOVIE = "data/derived/paradigm/paradigm_movie.mp4"

def frame(p, width=None, height=None):
    """Paradigm frame p as an RGB uint8 array (H, W, 3). ~17 ms (seeks to the preceding keyframe)."""
    with av.open(MOVIE) as c:
        s = c.streams.video[0]
        c.seek(int(p / paradigm.FPS / s.time_base), stream=s)
        for fr in c.decode(s):
            if round(fr.time * paradigm.FPS) == p:
                return fr.to_ndarray(width=width, height=height, format="rgb24")

def frames(first=0, last=paradigm.N_FRAMES - 1, width=None, height=None):
    """Yield (p, RGB array) for p = first … last in order. ~900 frames/s at 224×94."""
    with av.open(MOVIE) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        c.seek(int(first / paradigm.FPS / s.time_base), stream=s)
        for fr in c.decode(s):
            p = round(fr.time * paradigm.FPS)
            if p < first:
                continue
            if p > last:
                return
            yield p, fr.to_ndarray(width=width, height=height, format="rgb24")
```

- For features over the whole movie (CLIP, DINO, a VLM…), make **one sequential pass** with `frames()`;
  don't call `frame()` 125k times. Store results with **one row per paradigm frame** (or per bin, see §6)
  so they line up with everything else.
- Resize in `to_ndarray(width=…, height=…)`; it is faster than resizing in Python. Mind the 2.4:1 aspect
  if a model wants square input (pad or crop deliberately).
- **Quality flags**: `data/derived/alignment/frame_map.npy` has a `flag` per paradigm frame. 0 = exact
  (124,117 frames); 1 = may be off by one frame (1,611 frames around re-timing glitches in the source
  copy; harmless for most purposes); 2 = no picture (frames 125,728 – 125,742, end credits: black).
- **Never** read frames from the source file in `data/movie/` and index them by `p`: its timing
  is different (another frame rate, a 7-minute section the patients never saw). The same goes for the
  authors' `movie_wrapper` code; `docs/ALIGNMENT.md` explains why.
- To *look* at frames (sanity checks), save a few as PNG (`PIL.Image.fromarray(frame(p)).save(...)`)
  and view them.

## 3. Audio and text on the same clock

| File | Use |
|---|---|
| `data/derived/paradigm/audio_de_16k.flac` | analysis audio (German, mono, int16, 16 kHz). **Sample `n` is at `n / 16000` s; frame `p` = samples `[640 p, 640 (p+1))`**. Total 125,743 × 640 samples. |
| `data/derived/paradigm/audio_en_16k.flac` | same for the English track (not what patients heard) |
| `data/derived/paradigm/audio_{de,en}.m4a` | 48 kHz stereo, for listening |
| `data/derived/paradigm/subs_{de,en}.json` | `[{"start": s, "end": s, "text": …}]` in paradigm seconds (also `.srt`, `.vtt`) |

```python
a = np.concatenate([f.to_ndarray().ravel()
                    for f in av.open("data/derived/paradigm/audio_de_16k.flac").decode(audio=0)])
p = 9443                                     # any paradigm frame
chunk = a[640 * p: 640 * (p + 1)]           # the 40 ms of German audio under frame p
```

The audio runs 4.27 % faster than the film (PAL speed-up), so its pitch is slightly raised.
The subtitles are a subtitle file, not the dub script; for the words actually spoken, run speech
recognition on `audio_de_16k.flac` (the timestamps it returns are already paradigm time).

## 4. Annotations (same for every patient)

```python
from summer import nwb
labels = nwb.indicator_functions()      # {name: uint8 array (125,743,)}, element p = frame p
nwb.onsets("camera-cuts")               # paradigm frames where a label turns on
nwb.intervals("summer-faces")           # (n, 2) [first, last+1) frame ranges
```

53 labels, identical in all 29 NWB files (read from the first one and cached):

- **characters** (on screen): `tom`, `summer`, `mckenzie`, `rachel`, `paul`, `vance`, `autumn`, `alison`,
  `douche`, `millie`, `rhoda`, `secretary`
- **faces**: `tom-faces`, `summer-faces`, `mckenzie-faces`, `rachel-faces`, `paul-faces`, `vance-faces`, `millie-faces`
- **speech** (main characters' voices): `tom-speaking`, `summer-speaking`
- **presence**: `summer-presence`, `summer-body-sequence`, `persons`
- **transitions**: `camera-cuts` (the first frame of each new shot; 958), `scenes` (133), `days-of-summer`
  (the "(n) Days of Summer" title cards; 34)
- **setting**: `indoor-setting`, `the-graduate` (the film-within-the-film on TV)
- **24 locations**: `beach`, `bus`, `cafe`, `car`, `elevator`, `family-home`, `gallery`, `ikea`, `karaoke-bar`,
  `office`, `other-cafe`, `park`, `punch-bar`, `record-store`, `restaurant`, `soccer-field`, `street`,
  `summer-apartment`, `summer-child-bedroom`, `theater`, `tom-apartment`, `tom-child-bedroom`, `train`,
  `wedding-venue`

All are 0/1 **except `indoor-setting`: 0 = outdoor, 1 = indoor, 99 = undefined** (logos, title
cards). `onsets`/`intervals` treat only `== 1` as "on"; when you use the raw arrays, handle the 99s.
Many labels are rare (e.g. `rhoda` is 0.3 % of frames); check the base rate before training on one.

## 5. Neural data

```python
from summer import neural, nwb
from summer.paths import patients
patients()                                  # [1, 2, …, 29]
meta, trains = neural.spike_trains(14)      # one patient
# meta: dict of per-unit lists/arrays: unit_id, brain_region, hemisphere, is_single_unit, cell_type
# trains[i]: sorted spike times of unit i in paradigm seconds (float64)
```

- `spike_trains` maps each spike from the recording clock to movie time through that patient's watchlog,
  and **drops spikes that fell during playback pauses** (sessions have 0–11 pauses). Every session covers
  the analysis window (logs run to 4779 – 4804 s). Raw times: `nwb.units(p)["spike_times_ms"]`
  (recording ms, 0 = movie start). The mapping: `neural.neural_to_paradigm_time(p, t_ms)`.
- Units are different in every patient: pool them across patients as a *pseudo-population*.
- 2,286 units, 1,190 putative single units (`is_single_unit`). `cell_type`: pyramidal 1,087, multi-unit 1,096,
  interneuron 91, negative-peak 12. Hemisphere: L 1,178 / R 1,108.
- Regions (`brain_region`): `A` amygdala 559 · `EC` entorhinal cortex 493 · `AH` anterior hippocampus 418 ·
  `PH` posterior hippocampus 412 · `PHC` parahippocampal cortex 293 · `PIC` piriform cortex 68 ·
  `MH` medial hippocampus 16 · `LG` lingual gyrus 15 · `FF` fusiform gyrus 6 · `PRC` perirhinal cortex 6.
- Other unit columns (quality metrics, waveforms) are in the NWB file itself: `h5py.File(paths.nwb_path(p))["units"]`
  has `cv2`, `isi_violations`, `iso_dist`, `peak_SNR`, `waveform_mean`, `waveform_sem`, `csc_nr`.
  Electrode coordinates are in `general/extracellular_ephys/electrodes` (`x`, `y`, `z`, `location`).

## 6. Bins: spike counts and labels on the authors' grid

The NWB files provide bin edges for 40, 80, 200, 480 and 1000 ms over the analysis window
(`processing/machine_learning/movie_binning_info`). The authors suggest **80 ms** for prediction tasks.
Use these edges so results are comparable with the paper:

```python
import h5py, numpy as np
from summer import neural, nwb, paradigm
from summer.paths import nwb_path, patients

with h5py.File(nwb_path(1)) as f:                               # identical in every file
    bi = f["processing/machine_learning/movie_binning_info"]
    k = list(bi["bin_length"][:]).index(80)
    ends = bi["edges_index"][:]
    edges = bi["edges"][(ends[k - 1] if k else 0):ends[k]]     # (59081,) paradigm seconds, 36.72 … 4763.12
first_frame = np.round(edges[:-1] / paradigm.FRAME_S).astype(int)   # bin i starts at this frame (918, 920, …)

meta, trains = neural.spike_trains(14)
counts = np.stack([np.histogram(t, edges)[0] for t in trains], axis=1)  # (59080 bins, n_units)
y = nwb.indicator_functions()["summer-faces"][first_frame]              # label per bin = label of its first frame

# pseudo-population: amygdala single units from every patient -> (59080, 303)
cols = []
for pid in patients():
    m, tr = neural.spike_trains(pid)
    cols += [np.histogram(t, edges)[0]
             for t, region, su in zip(tr, m["brain_region"], m["is_single_unit"]) if region == "A" and su]
X = np.stack(cols, axis=1)
```

Bin `i` covers frames `first_frame[i] … first_frame[i] + bin_ms // 40 - 1` (2 frames per 80 ms bin), so an
image feature per bin is the mean over those frames' rows.

## 7. Train/test splits: never random

Neighbouring bins are almost identical (in both the movie and the firing), so random splits leak and
inflate results badly. Split the movie into **contiguous blocks, with a buffer of at least 32 s** between
training and evaluation data (the paper's recommendation). The authors' exact 5-fold scheme is
`SUMMERDataModule` in `ML_framework/dataloader.py` of https://github.com/mormannlab/SUMMER; copy it if
you compare numbers with the paper. Neuron responses lag the stimulus by a few hundred ms, so consider
predicting from lagged bins.

## 8. Where everything is

| Path | What |
|---|---|
| `data/nwb/sub-{p}/sub-{p}_ses-sub{p}_ecephys.nwb` | the raw sessions (29), read via `summer.nwb` |
| `data/derived/paradigm/paradigm_movie.mp4` | **the aligned frames** (+ DE/EN audio) |
| `data/derived/paradigm/audio_*`, `subs_*` | aligned audio and subtitles (§3) |
| `data/derived/alignment/frame_map.npy` | per-frame alignment quality flags (§2) |
| `data/derived/alignment/report.html`, `paradigm/report.html` | the alignment evidence |
| `data/derived/viewer/tracks/` | viewer data; add your results here (§9) |
| `data/movie/` | the user's source copy of the movie: **input only, don't index into it** |
| `src/summer/` | library + CLI; `docs/` for humans (`ALIGNMENT.md`, `DATA.md`, `TRACKS.md`) |

`data/` is git-ignored and the path can be changed in `summer.toml`, so build paths with
`summer.paths` (`paths.PARADIGM`, `paths.ALIGNMENT`, `paths.nwb_path(p)`, …) rather than hard-coding them.

## 9. Your own outputs

- Put experiment code in `experiments/<name>/` (plain scripts) and large outputs (features, models) in
  `data/derived/<name>/` (git-ignored). Index per-frame outputs by paradigm frame (`N_FRAMES` rows), and
  per-bin outputs by the bin grid above (save the edges with them).
- To show a result in the viewer next to the movie and spikes (e.g. VLM captions or a feature time series),
  write a track with `summer.tracks`; see `docs/TRACKS.md`. The user can then open it with `uv run summer view`
  (a browser app for people; there's no need to run it yourself).
- Don't modify `data/derived/paradigm/`, `alignment/` or `stamps/` by hand; they are built by `summer setup`.

## 10. Before you trust a result

- Did you use paradigm frames/seconds everywhere, and the 0-based index (§1)?
- Did you restrict to the analysis window, and handle `indoor-setting == 99`?
- Is your train/test split contiguous with a ≥ 32 s buffer (§7)?
- Is the base rate of your label reasonable, and is your metric above a shuffled/shifted-label control?
- For stimulus features: spot-check a few frames by saving them as images; check that `camera-cuts` onsets
  line up with big changes in your features.
