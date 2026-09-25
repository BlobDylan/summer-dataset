# summer-dataset

Tools for working with **SUMMER** (DANDI [001616](https://dandiarchive.org/dandiset/001616)): single-neuron
recordings from 29 patients (2,286 units, medial temporal lobe) while they watched *(500) Days of Summer*.
Paper: [Scientific Data 2026](https://www.nature.com/articles/s41597-026-07955-0) · authors' code:
[mormannlab/SUMMER](https://github.com/mormannlab/SUMMER).

This repo gets you from nothing to:

- the NWB files, downloaded and checksummed;
- **the movie exactly as the patients saw it**, rebuilt from your own copy and verified frame by frame:
  `paradigm_movie.mp4` where frame *i* is paradigm frame *i*, with German and English audio and
  subtitles on the same clock as every annotation and spike;
- a **viewer** that plays the movie alongside the annotations, subtitles and spike rasters of any
  patients/regions, and that you can add your own data tracks to.

## Setup

Needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Nothing else:
Python is fetched by uv, FFmpeg comes bundled with PyAV, no GPU.

```bash
git clone git@github.com:BlobDylan/summer-dataset.git && cd summer-dataset
uv sync
uv run summer download-nwb        # 2.4 GB from DANDI, SHA-256 verified
#   put the movie + subtitles in data/movie/  (lab: shared Drive folder, see docs/DATA.md)
uv run summer status              # checks the inputs, says what's next
uv run summer setup               # align → verify → build → verify again → viewer data (~20 min)
uv run summer view                # opens http://127.0.0.1:8765
```

`summer setup` only redoes what changed, and stops if the alignment checks fail. Open
`data/derived/alignment/report.html` once to see the evidence (contact sheets included).

## Using the data

Everything is on the **paradigm axis**: frame `p = 0 … 125,742` at 25 fps, `t = p × 0.04 s` — the axis of
the NWB annotations, bin edges and watchlogs.

```python
from summer import nwb, neural, paradigm

labels = nwb.indicator_functions()          # {"summer-faces": uint8[125743], ...}, index = p
meta, trains = neural.spike_trains(14)      # sub-14: spike times in paradigm seconds (pauses dropped)
paradigm.ANALYSIS_FIRST, paradigm.ANALYSIS_LAST   # 918, 119079: the authors' analysis window
```

- Frames: decode `data/derived/paradigm/paradigm_movie.mp4`; frame *i* is paradigm frame *i*.
- Audio for analysis: `data/derived/paradigm/audio_de_16k.flac`; frame `p` = samples `[640 p, 640 (p+1))`.
- Careful: `movie_binning_info` names frames 1-based (`frame_000919.jpg` is `p = 918`).

## Commands

| | |
|---|---|
| `summer download-nwb` | fetch/verify the NWB files |
| `summer status` | inputs found, steps done/stale/missing |
| `summer setup [--skip-video] [--force]` | run whatever is missing or out of date |
| `summer run <step>` | one step: `align`, `verify`, `audio`, `subs`, `movie`, `verify-built`, `viewer-data` |
| `summer probe <file>` | streams + fingerprint of a movie file |
| `summer register-copy "<desc>"` | add your (verified) movie copy to `known_copies.json` |
| `summer view [--port N]` | the viewer |

Viewer keys: space play/pause · `,` `.` frame step · `←` `→` ±5 s (shift: 30 s) · click the timeline to
seek · ctrl/pinch-scroll on the timeline to zoom.

## Docs

- [AGENTS.md](AGENTS.md): **start here for experiments** (also what AI agents read): the aligned frames, audio,
  labels, spikes, bins, splits and pitfalls, with tested code
- [docs/DATA.md](docs/DATA.md): inputs, where they go, other movie copies, outputs
- [docs/ALIGNMENT.md](docs/ALIGNMENT.md): conventions, the mapping, how it is measured and verified, known limits
- [docs/TRACKS.md](docs/TRACKS.md): adding your own data (e.g. VLM captions) to the viewer

## Layout

```
src/summer/
  paradigm.py    constants, index conventions, FilmMapping
  nwb.py         annotations, units, watchlogs (h5py)
  neural.py      spike times -> paradigm time
  movie.py       probing, fingerprints, frame signals, decoding
  align.py       measure the mapping          verify.py   checks + report
  build.py       paradigm movie/audio/subs    pipeline.py steps, stamps, status
  download.py    DANDI download               config.py   summer.toml
  tracks.py      write viewer tracks          viewer_data.py  default tracks
  server.py      local server                 viewer_app/     the viewer (plain JS)
tests/           conventions, NWB fixture, synthetic end-to-end alignment (run in CI)
```

`uv run pytest` runs the tests; they need no data.
