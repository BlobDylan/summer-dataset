"""summer: set up and explore the SUMMER dataset. Run `summer -h`."""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="summer", description="Set up and explore the SUMMER dataset.")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    sub.add_parser("download-nwb", help="download + checksum the 29 NWB files from DANDI (~2.4 GB)")
    p = sub.add_parser("probe", help="list the streams of a movie file (to fill in summer.toml)")
    p.add_argument("file", type=Path)
    sub.add_parser("status", help="what is present, what is built, what to do next")
    s = sub.add_parser("setup", help="run every step that is missing or out of date")
    s.add_argument("--skip-video", action="store_true", help="skip the paradigm movie (~12 min, ~2 GB)")
    s.add_argument("--force", action="store_true", help="re-run every step")
    r = sub.add_parser("run", help="run one pipeline step (normally `setup` does this)")
    r.add_argument("step", choices=["align", "verify", "audio", "subs", "movie", "verify-built", "viewer-data"])
    rc = sub.add_parser("register-copy", help="add your movie copy to known_copies.json")
    rc.add_argument("description")
    v = sub.add_parser("view", help="open the viewer")
    v.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()

    if a.cmd == "download-nwb":
        from . import download
        download.run()
    elif a.cmd == "probe":
        from . import movie
        info = movie.probe(a.file)
        print(json.dumps(info, indent=2))
        print(f"video fingerprint: {movie.fingerprint(a.file)}")
    elif a.cmd == "status":
        from . import pipeline
        pipeline.status()
    elif a.cmd == "setup":
        from . import pipeline
        pipeline.setup(skip_video=a.skip_video, force=a.force)
    elif a.cmd == "run":
        from . import pipeline
        inp = pipeline.check_inputs()
        pipeline.run_step(a.step, inp)
        pipeline._write_stamp(a.step, pipeline.signatures(inp)[a.step])
    elif a.cmd == "register-copy":
        from . import pipeline
        pipeline.register_copy(a.description)
    elif a.cmd == "view":
        from . import server
        server.run(a.port)


if __name__ == "__main__":
    main()
