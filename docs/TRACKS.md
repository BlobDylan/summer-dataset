# Viewer tracks

The viewer (`summer view`) draws **tracks**: one JSON file each in `data/derived/viewer/tracks/`, all
times in paradigm seconds (`p × 0.04`). It lists every file in that folder on page load, so adding data
to the viewer means writing a file and reloading.

## Writing a track

```python
from summer import tracks

# timed text: captions, VLM descriptions, transcripts
tracks.text("vlm_captions", "VLM captions", [(start_s, end_s, "a man sits on a bench"), ...],
            group="Text", description="Qwen2-VL, one caption per 2 s")

# a regularly sampled number: CLIP novelty, audio loudness, decoder output...
tracks.series("clip_novelty", "CLIP novelty", t0=0.0, dt=0.08, values=values, group="Stimulus", unit="1 - cos")

# on/off labels (grouped for the sidebar filter)
tracks.intervals("my_labels", "My labels", {"kiss": [(3120.4, 3124.0)], "rain": [...]},
                 group="Labels", label_groups={"Events": ["kiss", "rain"]})
```

Put per-frame values on the paradigm axis first: a frame-indexed array `v[p]` becomes
`tracks.series(..., t0=0.0, dt=0.04, values=v)`; bin values from `movie_binning_info` start at
`t0 = 36.72` with `dt` = the bin length.

Delete a track by deleting its file. `summer setup` rewrites only the default tracks (alignment,
annotations, subtitles, frame change, spikes) and leaves yours alone.

## Track types

| type | drawn as | extra |
|---|---|---|
| `intervals` | one lane per enabled label | label checkboxes by group; active labels appear as chips under the video |
| `text` | boxes with text | can be overlaid on the video (overlay menu); current text appears under the video |
| `series` | line (min/max per pixel when zoomed out) | y-range from 0 (or the minimum, if negative) to the 99.5th percentile |
| `spikes` | raster or rate heatmap + population rate | patient / region / single-unit filters |

A new *kind* of visualisation needs a renderer in `src/summer/viewer_app/app.js`: add an entry to
`RENDERERS` with `init(track)` (returns sidebar controls or null), `height(track)`, `draw(track, g, y, t0, t1)`,
and optionally `now(track, t)` and `hover(track, t, dy)`. The existing four are the templates.
`window.viewer` exposes the viewer's state in the browser console.
