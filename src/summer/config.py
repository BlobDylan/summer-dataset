"""Where the inputs live: summer.toml in the repo root (optional; see summer.example.toml)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "summer.toml"


@dataclass(frozen=True)
class Source:
    """One audio track: a file and the stream index inside it (None = pick by language tag)."""
    file: Path
    stream: int | None


@dataclass(frozen=True)
class Config:
    data_dir: Path
    video: Path
    audio: dict[str, Source]                       # "de" (required), "en" (optional)
    subtitles: dict[str, Path] = field(default_factory=dict)


def _parse_source(s: str, base: Path) -> Source:
    path, _, idx = s.partition("#")
    return Source(base / path, int(idx) if idx else None)


def load() -> Config:
    raw = tomllib.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
    data_dir = ROOT / raw.get("data_dir", "data")
    movie = raw.get("movie", {})
    video = data_dir / movie.get("video", "movie/movie-eng.mp4")
    audio_raw = movie.get("audio", {"de": "movie/movie-eng.mp4", "en": "movie/movie-eng.mp4"})
    subs_raw = movie.get("subtitles", {"de": "movie/movie-subs-de", "en": "movie/movie-subs-eng"})
    return Config(
        data_dir=data_dir,
        video=video,
        audio={lang: _parse_source(s, data_dir) for lang, s in audio_raw.items()},
        subtitles={lang: data_dir / p for lang, p in subs_raw.items()},
    )
