import os
from typing import cast

from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]


class VideoDownloadError(Exception):
    pass


def download_video(url: str, dest_dir: str) -> str:
    """Downloads `url` into `dest_dir` using yt-dlp and returns the local file path."""
    options = {
        "outtmpl": os.path.join(dest_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            return cast(str, ydl.prepare_filename(info))
    except DownloadError as e:
        raise VideoDownloadError(str(e)) from e
