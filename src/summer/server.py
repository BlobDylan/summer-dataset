"""Tiny local server for the viewer: static app + data files with HTTP Range (needed for seeking)."""

import json
import mimetypes
import re
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import paradigm
from .paths import DERIVED, PARADIGM, VIEWER_DATA

APP = Path(__file__).parent / "viewer_app"
mimetypes.add_type("text/vtt", ".vtt")
mimetypes.add_type("audio/mp4", ".m4a")


def manifest() -> dict:
    tracks = []
    for f in sorted((VIEWER_DATA / "tracks").glob("*.json")):
        with open(f) as fh:
            head = json.load(fh)
        tracks.append({k: head.get(k) for k in ("id", "name", "type", "group", "description")}
                      | {"url": f"/data/viewer/tracks/{f.name}"})
    return dict(
        fps=paradigm.FPS, n_frames=paradigm.N_FRAMES,
        analysis=[paradigm.ANALYSIS_FIRST, paradigm.ANALYSIS_LAST],
        video="/data/paradigm/paradigm_movie.mp4",   # played muted; audio comes from the m4a files
        audio={lang: f"/data/paradigm/audio_{lang}.m4a" for lang in ("de", "en")
               if (PARADIGM / f"audio_{lang}.m4a").exists()},
        tracks=tracks,
    )


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/manifest":
            body = json.dumps(manifest()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/data/"):
            f = (DERIVED / path[len("/data/"):]).resolve()
            root = DERIVED.resolve()
        else:
            f = (APP / (path.lstrip("/") or "index.html")).resolve()
            root = APP.resolve()
        if not f.is_file() or root not in f.parents:
            self.send_error(404)
            return
        self._send_file(f)

    def _send_file(self, f: Path):
        size = f.stat().st_size
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        start, end = 0, size - 1
        if m:
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = size - int(m.group(2))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass   # the browser cancels range requests while seeking; that's normal


def run(port: int):
    assert (PARADIGM / "paradigm_movie.mp4").exists(), "run `summer setup` first (without --skip-video)"
    assert (VIEWER_DATA / "tracks").exists(), "run `summer setup` first"
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"viewer at {url}  (Ctrl-C to stop)")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
