import os
import uuid
from typing import cast

from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import YoutubeDLError  # type: ignore[import-untyped]


class VideoDownloadError(Exception):
    pass


def download_video(url: str, dest_dir: str) -> str:
    """Downloads `url` into `dest_dir` using yt-dlp and returns the local file path."""
    # A random name we fully control, instead of yt-dlp's own %(id)s: we
    # download the Scrapper Backend's videoUrl directly (a raw CDN link, not
    # the platform's own webpage URL), so no site-specific extractor
    # recognizes it and yt-dlp falls back to its generic extractor, which
    # derives %(id)s from the URL itself. For some CDNs (observed for
    # TikTok's) that fallback id is the entire query string -- 300+
    # characters, past Windows' MAX_PATH even after sanitizing unsafe
    # characters (see docs/issue-findings.md). Never let the destination
    # filename depend on extractor-provided metadata.
    basename = uuid.uuid4().hex
    options = {
        "outtmpl": os.path.join(dest_dir, f"{basename}.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        # A single progressive format avoids needing to merge separate
        # video+audio formats (which would need ffmpeg muxing anyway).
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
