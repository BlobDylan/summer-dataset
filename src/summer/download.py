"""Download the SUMMER NWB files from DANDI (public, no account needed) and verify their SHA-256."""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .paths import NWB_DIR

DANDISET = "001616"
VERSION = "0.260702.0824"          # the published version cited by the paper (29 sessions)
API = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}"


def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def list_assets() -> list[dict]:
    assets, url = [], f"{API}/assets/?page_size=100"
    while url:
        page = _get(url)
        assets += page["results"]
        url = page["next"]
    for a in assets:
        a["sha256"] = _get(f"{API}/assets/{a['asset_id']}/")["digest"]["dandi:sha2-256"]
    return assets


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _fetch(a: dict) -> str:
    dest = NWB_DIR / a["path"]
    if dest.exists() and dest.stat().st_size == a["size"] and _sha256(dest) == a["sha256"]:
        return f"ok        {a['path']}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(f"{API}/assets/{a['asset_id']}/download/", tmp)
    if _sha256(tmp) != a["sha256"]:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {a['path']}; try again")
    tmp.replace(dest)
    return f"download  {a['path']}  ({a['size'] / 1e6:.0f} MB)"


def run(workers: int = 4):
    print(f"DANDI {DANDISET} version {VERSION}: listing assets…")
    assets = list_assets()
    print(f"{len(assets)} files, {sum(a['size'] for a in assets) / 1e9:.2f} GB -> {NWB_DIR}")
    with ThreadPoolExecutor(workers) as pool:
        for line in pool.map(_fetch, assets):
            print(" ", line)
    (NWB_DIR / "assets.json").write_text(json.dumps(assets, indent=1))
    print("all files present and verified")
