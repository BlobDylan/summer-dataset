"""Download the actual NWB files for dandiset 001616 listed in data/assets.json.

Each asset's DANDI "download" endpoint 302-redirects to a presigned S3 URL
on a public bucket, so no auth/API key is needed. Run scripts/list_assets.py
first to generate data/assets.json.
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DANDISET_ID = "001616"
ASSETS_PATH = Path(__file__).parent.parent / "data" / "assets.json"
OUT_DIR = Path(__file__).parent.parent / "data" / "nwb"
MAX_WORKERS = 4


def download_url(asset_id: str) -> str:
    url = (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}"
        f"/versions/draft/assets/{asset_id}/download/"
    )
    return url


def download_asset(asset: dict) -> None:
    dest = OUT_DIR / asset["path"]
    if dest.exists() and dest.stat().st_size == asset["size"]:
        print(f"Skipping {asset['path']} (already downloaded)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {asset['path']} ({asset['size'] / 1e9:.2f} GB)...")
    urllib.request.urlretrieve(download_url(asset["asset_id"]), dest)
    print(f"Finished {asset['path']}")


def main() -> None:
    assets = json.loads(ASSETS_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(download_asset, assets))

    print("Done.")


if __name__ == "__main__":
    main()
