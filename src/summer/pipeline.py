"""The pipeline: input checks, steps with up-to-date stamps, `status` and `setup`.

Each step records a *signature* of what it was built from (input fingerprints, mapping, code
version) in data/derived/stamps/<step>.json. A step is up to date when its signature is
unchanged and its outputs exist, so `summer setup` only redoes what changed.
"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import align, build, movie, paradigm, verify
from .paths import ALIGNMENT, CONFIG, KNOWN_COPIES, NWB_DIR, PARADIGM, STAMPS, VIEWER_DATA, patients

N_SESSIONS = 29
MOVIE = PARADIGM / "paradigm_movie.mp4"


# --- inputs -----------------------------------------------------------------------------------

def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def known_copies() -> dict:
    return json.loads(KNOWN_COPIES.read_text()) if KNOWN_COPIES.exists() else {}


def check_inputs(strict: bool = True) -> dict:
    """Inspect the configured inputs. Returns a report; with strict=True exits on a problem."""
    rep = {"problems": []}
    n_nwb = len([p for p in patients() if (NWB_DIR / f"sub-{p}").exists()])
    rep["nwb"] = n_nwb
    if n_nwb < N_SESSIONS:
        rep["problems"].append(f"NWB: {n_nwb}/{N_SESSIONS} sessions in {NWB_DIR} -> run `summer download-nwb`")
    v = CONFIG.video
    if not v.exists():
        rep["problems"].append(f"movie: {v} not found -> see docs/DATA.md (or set [movie] video in summer.toml)")
    else:
        rep["video_fp"] = movie.fingerprint(v)
        rep["known_copy"] = known_copies().get(rep["video_fp"], {}).get("description")
        rep["audio"] = {}
        for lang, src in CONFIG.audio.items():
            if not src.file.exists():
                rep["problems"].append(f"audio {lang}: {src.file} not found")
                continue
            if movie.fingerprint(src.file) != rep["video_fp"]:
                rep["problems"].append(
                    f"audio {lang}: {src.file.name} does not contain the same video as {v.name}, so its "
                    f"timing can't be trusted. Use a file with identical video (see docs/DATA.md).")
                continue
            rep["audio"][lang] = (src.file, movie.audio_stream_index(src.file, src.stream, lang))
        if "de" not in rep["audio"] and not any(p.startswith("audio de") for p in rep["problems"]):
            rep["problems"].append("audio de: required (it is what the patients heard)")
    rep["subtitles"] = {lang: p for lang, p in CONFIG.subtitles.items() if p.exists()}
    if strict and rep["problems"]:
        raise SystemExit("Inputs are not ready:\n  " + "\n  ".join(rep["problems"]))
    return rep


# --- steps ------------------------------------------------------------------------------------

def _mapping_sig():
    p = ALIGNMENT / "alignment.json"
    return json.loads(p.read_text())["mapping"] if p.exists() else None


def signatures(inp: dict) -> dict:
    code = version("summer-dataset")
    audio = {lang: [movie.fingerprint(f), s] for lang, (f, s) in inp["audio"].items()}
    subs = {lang: _sha1(p) for lang, p in inp["subtitles"].items()}
    m = _mapping_sig()
    return {
        "align": dict(video=inp["video_fp"], code=code),
        "verify": dict(video=inp["video_fp"], mapping=m, audio_de=audio.get("de"), code=code),
        "audio": dict(mapping=m, audio=audio, code=code),
        "subs": dict(mapping=m, subs=subs, code=code),
        "movie": dict(video=inp["video_fp"], mapping=m, audio=audio, code=code),
        "verify-built": dict(video=inp["video_fp"], mapping=m, audio=audio, code=code),
        "viewer-data": dict(mapping=m, subs=subs, nwb=len(patients()), code=code),
    }


OUTPUTS = {
    "align": [ALIGNMENT / "alignment.json", ALIGNMENT / "frame_map.npy"],
    "verify": [ALIGNMENT / "verification.json", ALIGNMENT / "report.html"],
    "audio": [PARADIGM / "audio_de.m4a", PARADIGM / "audio_de_16k.flac"],
    "subs": [],
    "movie": [MOVIE],
    "verify-built": [PARADIGM / "verification.json"],
    "viewer-data": [VIEWER_DATA / "tracks" / "annotations.json"],
}
ORDER = ["align", "verify", "audio", "subs", "movie", "verify-built", "viewer-data"]


def _stamp(step):
    p = STAMPS / f"{step}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _write_stamp(step, sig):
    STAMPS.mkdir(parents=True, exist_ok=True)
    (STAMPS / f"{step}.json").write_text(json.dumps(sig, indent=1))


def state(step, sigs) -> str:
    s = _stamp(step)
    if s is None or not all(p.exists() for p in OUTPUTS[step]):
        return "missing"
    return "ok" if s == sigs[step] else "stale"


def _passed(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text()).get("pass", False)


def run_step(step: str, inp: dict):
    mp = align.load()[0] if step != "align" else None
    if step == "align":
        align.run(CONFIG.video)
        _report_known_copy(inp["video_fp"])
    elif step == "verify":
        f, s = inp["audio"]["de"]
        verify.run(CONFIG.video, mp, audio_stream=s, out_dir=ALIGNMENT, title="Source movie vs paradigm",
                   audio_source=f)
        if not _passed(ALIGNMENT / "verification.json"):
            raise SystemExit(f"Verification FAILED - see {ALIGNMENT / 'report.html'}. Not building on it.")
    elif step == "audio":
        PARADIGM.mkdir(parents=True, exist_ok=True)
        for lang, (f, s) in inp["audio"].items():
            build.build_audio(f, s, mp, lang)
    elif step == "subs":
        PARADIGM.mkdir(parents=True, exist_ok=True)
        for lang, p in inp["subtitles"].items():
            build.build_subtitles(mp, p, lang)
    elif step == "movie":
        # re-encode the video only if the video/mapping changed; otherwise just re-mux new audio
        prev = _stamp("movie")
        need_video = not MOVIE.exists() or prev is None or \
            (prev["video"], prev["mapping"], prev["code"]) != (inp["video_fp"], _mapping_sig(), version("summer-dataset"))
        video_src = MOVIE
        if need_video:
            video_src = PARADIGM / "video.tmp.mp4"
            build.build_video(CONFIG.video, np.load(ALIGNMENT / "frame_map.npy"), video_src)
        build.mux(video_src, [(PARADIGM / f"audio_{lang}.m4a", *build.LANGS[lang]) for lang in inp["audio"]], MOVIE)
        if need_video:
            video_src.unlink()
    elif step == "verify-built":
        verify.run(MOVIE, paradigm.IDENTITY, audio_stream=1, out_dir=PARADIGM,
                   title="Built paradigm movie (identity mapping)")
        if not _passed(PARADIGM / "verification.json"):
            raise SystemExit(f"Round-trip verification FAILED - see {PARADIGM / 'report.html'}")
    elif step == "viewer-data":
        from . import viewer_data
        viewer_data.run()


def setup(skip_video: bool = False, force: bool = False):
    inp = check_inputs()
    for step in ORDER:
        if skip_video and step in ("movie", "verify-built"):
            continue
        sigs = signatures(inp)             # recomputed: the mapping may have just changed
        st = state(step, sigs)
        if st == "ok" and not force:
            print(f"[{step}] up to date")
            continue
        print(f"[{step}] {'running' if st == 'missing' else 'inputs changed, re-running'}")
        run_step(step, inp)
        _write_stamp(step, signatures(inp)[step])
    print("\nDone. `summer view` opens the viewer.")


def status():
    inp = check_inputs(strict=False)
    print("Inputs")
    print(f"  NWB sessions     {inp['nwb']}/{N_SESSIONS}  ({NWB_DIR})")
    if "video_fp" in inp:
        kc = f"known copy: {inp['known_copy']}" if inp["known_copy"] else "not in known_copies.json (will be measured)"
        print(f"  movie            {CONFIG.video}  [{inp['video_fp']}]  {kc}")
        for lang, (f, s) in inp["audio"].items():
            print(f"  audio {lang}         {f.name} stream #{s}")
    for lang in CONFIG.subtitles:
        p = CONFIG.subtitles[lang]
        print(f"  subtitles {lang}     {'found' if p.exists() else 'not found (optional)'}  {p}")
    for pr in inp["problems"]:
        print(f"  ! {pr}")
    if inp["problems"]:
        return
    print("Steps")
    sigs = signatures(inp)
    for step in ORDER:
        print(f"  {step:13s} {state(step, sigs)}")
    todo = [s for s in ORDER if state(s, sigs) != "ok"]
    print(f"\nNext: {'`summer setup`' if todo else '`summer view`'}")


# --- known copies -----------------------------------------------------------------------------

def _report_known_copy(fp: str):
    kc = known_copies().get(fp)
    mp = _mapping_sig()
    if kc is None:
        print("This copy is not in known_copies.json. After `summer verify` passes you can add it with "
              "`summer register-copy \"<description>\"`.")
    elif kc["mapping"] == mp:
        print(f"Known copy ({kc['description']}): measured mapping matches the registry.")
    else:
        print(f"WARNING: known copy ({kc['description']}) but the measured mapping differs from the registry:\n"
              f"  registry {kc['mapping']}\n  measured {mp}")


def register_copy(description: str):
    inp = check_inputs()
    ver = ALIGNMENT / "verification.json"
    if not _passed(ver):
        raise SystemExit("run `summer setup` (verification must pass) before registering a copy")
    v = json.loads(ver.read_text())
    info = json.loads((ALIGNMENT / "alignment.json").read_text())
    reg = known_copies()
    reg[inp["video_fp"]] = dict(
        description=description,
        video=dict((k, info["source"]["video"][k]) for k in ("codec", "width", "height", "avg_rate", "frames")),
        audio=[a["language"] for a in info["source"]["audio"]],
        mapping=info["mapping"],
        verification=dict(cuts_exact={k: round(x["exact"], 3) for k, x in v["cuts"]["mapping"].items()},
                          speech_offset_shift_s=round(v["audio"]["median_offset_shift_s"], 3)),
    )
    KNOWN_COPIES.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"registered {inp['video_fp']} in {KNOWN_COPIES.name}; commit it so colleagues see it")
