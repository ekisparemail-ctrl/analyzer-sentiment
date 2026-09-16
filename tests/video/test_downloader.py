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


def test_download_video_requests_a_single_progressive_format_with_safe_filenames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_ydl_instance = MagicMock()
    fake_ydl_instance.__enter__.return_value = fake_ydl_instance
    fake_ydl_instance.__exit__.return_value = False
    fake_ydl_instance.extract_info.return_value = {"id": "abc123", "ext": "mp4"}
    fake_ydl_instance.prepare_filename.return_value = str(tmp_path / "abc123.mp4")

    fake_ydl_class = MagicMock(return_value=fake_ydl_instance)
    monkeypatch.setattr("video.downloader.YoutubeDL", fake_ydl_class)

    download_video("https://tiktok.com/@user/video/1", str(tmp_path))

    options = fake_ydl_class.call_args[0][0]
    assert options["format"] == "best[ext=mp4]/best"
    assert options["restrictfilenames"] is True


def test_download_video_names_the_output_file_itself_instead_of_using_extracted_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression test: a real TikTok video (docs/issue-findings.md) failed
    # with "unable to open for writing: [Errno 22] Invalid argument". Root
    # cause: we download the Scrapper Backend's videoUrl directly (a raw CDN
    # link, not the platform's own webpage URL), so no site-specific
    # extractor recognizes it and yt-dlp falls back to its generic
    # extractor, which derives %(id)s from the URL itself -- for TikTok's
    # CDN that fallback id was the entire query string (300+ characters),
    # past Windows' MAX_PATH even after restrictfilenames sanitized unsafe
    # characters. The fix: never let the destination filename depend on
    # extractor-provided metadata -- generate it ourselves instead.
    fake_uuid = MagicMock()
    fake_uuid.hex = "deadbeefdeadbeefdeadbeefdeadbeef"
    monkeypatch.setattr("video.downloader.uuid.uuid4", lambda: fake_uuid)

    fake_ydl_instance = MagicMock()
    fake_ydl_instance.__enter__.return_value = fake_ydl_instance
    fake_ydl_instance.__exit__.return_value = False
    # A pathologically long id, as yt-dlp's generic extractor produced for
    # the real failing CDN URL -- must never end up in our outtmpl.
    fake_ydl_instance.extract_info.return_value = {"id": "a=1233&bti=..." * 20, "ext": "mp4"}
    fake_ydl_instance.prepare_filename.return_value = str(
        tmp_path / "deadbeefdeadbeefdeadbeefdeadbeef.mp4"
    )

    fake_ydl_class = MagicMock(return_value=fake_ydl_instance)
    monkeypatch.setattr("video.downloader.YoutubeDL", fake_ydl_class)

    download_video("https://v16m.tiktokcdn-us.com/very/long/query?a=1&b=2", str(tmp_path))

    options = fake_ydl_class.call_args[0][0]
    assert "deadbeefdeadbeefdeadbeefdeadbeef.%(ext)s" in options["outtmpl"]
    assert "%(id)s" not in options["outtmpl"]


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
