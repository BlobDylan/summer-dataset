# Data: what you need and where it goes

```
data/
  nwb/sub-*/sub-*_ses-sub*_ecephys.nwb   ← `summer download-nwb` (public, from DANDI)
  movie/movie-eng.mp4                    ← your copy of the movie (see below)
  movie/movie-subs-de, movie-subs-eng    ← optional subtitles (.srt)
  derived/                               ← everything `summer setup` makes
```

`data/` is git-ignored. To keep the data elsewhere (external drive), set `data_dir` in `summer.toml`
(copy `summer.example.toml`).

## 1. Neural data (NWB): automatic

`summer download-nwb` fetches the 29 sessions (2.4 GB) of DANDI [001616](https://dandiarchive.org/dandiset/001616),
version `0.260702.0824` (the one the paper cites), and checks every file's SHA-256. No account needed.
Re-running it only fetches what is missing or corrupt.

## 2. The movie: bring your own copy

The movie (*(500) Days of Summer*, 2009) is copyrighted and not part of the dataset. Lab members:
the copy this repo was set up with is in the shared Google Drive folder (ask Dylan for access) —
put `movie-eng.mp4` and the two subtitle files in `data/movie/`. `movie-de.mp4` from that folder is not
needed: its picture and German track are identical to `movie-eng.mp4`'s.

Any other copy should work too, as long as it:

- contains the **German** dub (what the patients heard) — English is optional, for your convenience;
- is the theatrical cut (≈ 95 min at film speed or ≈ 91 min PAL) in a format FFmpeg reads (mp4, mkv, …).

`summer setup` measures how your copy maps onto the version shown to patients and refuses to build
anything unless the checks pass (see [ALIGNMENT.md](ALIGNMENT.md)). If your copy's fingerprint is in
`known_copies.json`, you will see "known copy" and can compare the measured numbers against it.

### When the audio isn't where the defaults expect it

`summer probe data/movie/yourfile.mkv` lists the streams and their language tags. Then set them in
`summer.toml`:

```toml
[movie]
video = "movie/yourfile.mkv"
[movie.audio]
de = "movie/yourfile.mkv#2"      # stream index from `summer probe`
```

The German audio may come from a **different file** only if that file contains the *same video*
(same fingerprint in `summer probe`); otherwise its timing could differ and `summer setup` stops.

### Subtitles

Optional. They must be timed to your movie file (not to some other release). `summer setup` converts
them to paradigm time; the viewer shows them as tracks and as an overlay.

## 3. Outputs (`data/derived/`)

| Path | What |
|---|---|
| `alignment/alignment.json` | measured mapping, your copy's fingerprint, fit diagnostics |
| `alignment/frame_map.npy` | per paradigm frame: source frame index, source time, flag (0 exact, 1 may be ±1 frame, 2 beyond the source's end) |
| `alignment/report.html` | verification with controls and contact sheets — look at it once |
| `paradigm/paradigm_movie.mp4` | **frame i = paradigm frame i**, 25 fps; German (default) + English audio |
| `paradigm/audio_{de,en}.m4a` | 48 kHz stereo AAC on paradigm time |
| `paradigm/audio_{de,en}_16k.flac` | 16 kHz mono for analysis: sample `n` ↔ `n / 16000` s; frame `p` = samples `[640 p, 640 (p+1))` |
| `paradigm/subs_{de,en}.{srt,vtt,json}` | subtitles on paradigm time |
| `paradigm/report.html` | the same checks run on the built movie (round trip) |
| `viewer/tracks/` | the viewer's data tracks ([TRACKS.md](TRACKS.md)) |
| `stamps/`, `cache/` | bookkeeping so `summer setup` only redoes what changed |

Sizes: about 2.3 GB derived, most of it the movie. `summer setup --skip-video` makes everything except
the movie (and so the viewer) in a few minutes.
