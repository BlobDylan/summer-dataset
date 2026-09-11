# summer-dataset

Playground for DANDI dandiset [001616](https://dandiarchive.org/dandiset/001616) (SUMMER: human single-neuron activity during an 83-minute movie, 2,286 neurons, 29 patients).

## Getting the data

```
python scripts/list_assets.py     # fetch asset metadata -> data/assets.json
python scripts/download_data.py   # download NWB files -> data/nwb/
```

Data files are downloaded via DANDI's public API, which redirects to presigned S3 URLs on a public bucket — no auth needed. Downloaded files live under `data/`, which is gitignored.
