from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video.downloader import VideoDownloadError, download_video


def test_download_video_returns_path_of_downloaded_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written_file = tmp_path / "abc123.mp4"

    def fake_download(self: object, urls: list[str]) -> None:
        written_file.write_bytes(b"fake video bytes")

    def fake_prepare_filename(self: object, info: dict) -> str:
        return str(written_file)

    fake_ydl_instance = MagicMock()
    fake_ydl_instance.__enter__.return_value = fake_ydl_instance
    fake_ydl_instance.__exit__.return_value = False
    fake_ydl_instance.extract_info.return_value = {"id": "abc123", "ext": "mp4"}
    fake_ydl_instance.prepare_filename.return_value = str(written_file)

    fake_ydl_class = MagicMock(return_value=fake_ydl_instance)
    monkeypatch.setattr("video.downloader.YoutubeDL", fake_ydl_class)

    result_path = download_video("https://twitter.com/x/status/1", str(tmp_path))

    assert result_path == str(written_file)
    fake_ydl_instance.extract_info.assert_called_once_with(
        "https://twitter.com/x/status/1", download=True
    )


def test_download_video_wraps_extraction_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yt_dlp.utils import DownloadError

    fake_ydl_instance = MagicMock()
    fake_ydl_instance.__enter__.return_value = fake_ydl_instance
    fake_ydl_instance.__exit__.return_value = False
    fake_ydl_instance.extract_info.side_effect = DownloadError("video unavailable")

    fake_ydl_class = MagicMock(return_value=fake_ydl_instance)
    monkeypatch.setattr("video.downloader.YoutubeDL", fake_ydl_class)

    with pytest.raises(VideoDownloadError, match="video unavailable"):
        download_video("https://twitter.com/x/status/1", str(tmp_path))
