from pathlib import Path

from .config import ROOT, load

CONFIG = load()

DATA = CONFIG.data_dir
NWB_DIR = DATA / "nwb"
DERIVED = DATA / "derived"
CACHE = DERIVED / "cache"
ALIGNMENT = DERIVED / "alignment"
PARADIGM = DERIVED / "paradigm"                # the reconstructed paradigm movie + audio + subs
VIEWER_DATA = DERIVED / "viewer"
STAMPS = DERIVED / "stamps"                    # what each pipeline step was built from
KNOWN_COPIES = ROOT / "known_copies.json"


def nwb_path(patient: int) -> Path:
    return NWB_DIR / f"sub-{patient}" / f"sub-{patient}_ses-sub{patient}_ecephys.nwb"


def patients() -> list[int]:
    return sorted(int(p.name.split("-")[1]) for p in NWB_DIR.glob("sub-*") if p.is_dir())
