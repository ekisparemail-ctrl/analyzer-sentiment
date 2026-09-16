import os
from typing import cast

from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import YoutubeDLError  # type: ignore[import-untyped]


class VideoDownloadError(Exception):
    pass


def download_video(url: str, dest_dir: str) -> str:
    """Downloads `url` into `dest_dir` using yt-dlp and returns the local file path."""
    options = {
        "outtmpl": os.path.join(dest_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        # A real TikTok video hit "unable to open for writing: [Errno 22]
        # Invalid argument" -- a known yt-dlp/TikTok-extractor quirk where an
        # intermediate download temp-file gets named from a raw CDN URL
        # (query string included) instead of a sanitized filename, when
        # yt-dlp has to merge separate video+audio formats. A single
        # progressive format avoids that merge path entirely.
        "format": "best[ext=mp4]/best",
        # Defense in depth: force ASCII-safe filenames regardless of what
        # any title/description-derived filename component contains.
        "restrictfilenames": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            return cast(str, ydl.prepare_filename(info))
    except YoutubeDLError as e:
        raise VideoDownloadError(str(e)) from e
