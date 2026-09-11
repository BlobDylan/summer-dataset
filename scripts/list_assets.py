"""List assets for DANDI dandiset 001616 and save their metadata locally.

DANDI's own API (not the access-summaries repo, which only holds usage
stats) is what actually exposes per-asset info and no-auth S3 download
links, so that's what this hits.
"""

import json
import urllib.request
from pathlib import Path

DANDISET_ID = "001616"
API_URL = f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/versions/draft/assets/"
OUT_PATH = Path(__file__).parent.parent / "data" / "assets.json"


def fetch_all_assets() -> list[dict]:
    assets = []
    url = API_URL
    while url:
        with urllib.request.urlopen(url) as response:
            page = json.load(response)
        assets.extend(page["results"])
        url = page["next"]
    return assets


def main() -> None:
    assets = fetch_all_assets()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(assets, indent=2))
    print(f"Saved {len(assets)} asset records to {OUT_PATH}")


if __name__ == "__main__":
    main()
