"""
Dev-only tool: runs the full Analytics Backend pipeline (download, VLM,
whisper, LLM) against a hardcoded Scrapper Backend payload, bypassing Kafka
entirely.

Use this when the Scrapper Backend can't produce live traffic (e.g. its own
AI credits run out again) but the pipeline still needs to be exercised
end-to-end against a real message. Swap SAMPLE_PAYLOAD for whatever message
you need to replay -- paste the exact JSON as seen in Kafdrop/Kafka, camelCase
and all (it goes through the same NormalizedData parsing real Kafka messages
do). One payload can yield multiple analyzable items: the post itself plus
one per entry in its "comments" array, each analyzed and printed in turn.

Does NOT publish the result to the real analysis-to-scrapper topic -- it
only prints it, so ad-hoc replays never pollute the real result stream.

Usage:
    .venv/Scripts/python scripts/replay_sample_payload.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Real analyzed content (comments especially) routinely contains emoji and
# other characters outside Windows' legacy console codepage (cp1252) --
# without this, printing such a result crashes with UnicodeEncodeError after
# the item has already been fully processed (see docs/issue-findings.md).
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import Settings  # noqa: E402
from main import _dump_result_for_dev, build_dependencies  # noqa: E402
from messaging.scrapper_dto import NormalizedData, requests_from_normalized_data  # noqa: E402
from pipeline.analyze import analyze  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Real message captured from the scrapper-to-analysis topic (docs/issue-findings.md,
# 2026-09-17) -- includes one nested comment, so a replay run also exercises
# the per-comment analysis path alongside the post itself. NOTE: the videoUrl
# below is a signed TikTok CDN link with a short-lived token -- if this
# script starts getting HTTP 403 on download again, the token has expired
# and this payload needs replacing with a freshly captured one (see
# docs/issue-findings.md for the pattern this keeps hitting).
SAMPLE_PAYLOAD = {
    "id": "7685921933191400705",
    "platform": "tiktok",
    "message": (
        "Komisi Pemberantasan Korupsi menetapkan delapan orang tersangka dalam kasus "
        "dugaan korupsi pengurusan hak guna bangunan atau HGB di lingkungan Kementerian "
        "Agraria dan Tata Ruang/Badan Pertanahan Nasional (ATR/BPN), hasil operasi "
        "tangkap tangan di sejumlah lokasi. KPK menyita barang bukti uang mencapai "
        "lebih dari 106 miliar rupiah. #liputan6sctv #newssctv #emtekmedianews #kpk "
        "#korupsi"
    ),
    "url": "https://www.tiktok.com/@liputan6/video/7685921933191400705",
    "videoUrl": (
        "https://v77.tiktokcdn.com/bf4562d5c9dcaed3a935cc729f174fe0/6aacee16/video/tos/"
        "alisg/tos-alisg-pve-0037/ocAXMK4RAiEYus1p3aBBYVPgyl6qQnAjAi0Iz/"
        "?a=1233&bti=NEBzNTY6QGo6OjZALnAjNDQuYCMxNDNg&&bt=857"
        "&ft=EKpu4FZ-00gA12NvoMwnzIxRlbVclQ_45SY&mime_type=video_mp4"
        "&rc=NzY7ZTs7O2ZlZDk4NGlkO0BpM25manQ5cnF4ZDMzODgzNEAyXjY2YS8wXzMxYy40MC0wYSMyLmJu"
        "MmQ0LmhhLS1kLzFzcw%3D%3D&vvpl=1&l=2026091715511232142DF53097B7C6ED8A&btag=e00090000"
    ),
    "imageUrl": None,
    "authorUsername": "liputan6",
    "authorName": "Liputan6",
    "views": 544455,
    "likes": 10890,
    "repliesCount": 920,
    "uploadedAt": "2026-09-16T00:22:51.000Z",
    "comments": [
        {
            "id": "7685923821383566101",
            "message": "prabowo\U0001f970",
            "url": None,
            "videoUrl": None,
            "imageUrl": None,
            "authorUsername": "reserse61",
            "authorName": None,
            "likes": 283,
            "repliesCount": 15,
            "uploadedAt": "2026-09-16T00:30:12.000Z",
            "commentTo": "7685921933191400705",
        }
    ],
}


logger = logging.getLogger(__name__)


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    deps = build_dependencies(settings)

    # One message now yields the post itself plus one request per nested
    # comment (SAMPLE_PAYLOAD["comments"]) -- analyze and print each.
    requests = requests_from_normalized_data(NormalizedData(**SAMPLE_PAYLOAD))
    for request in requests:
        result = analyze(request, deps)

        # Timestamped summary in the log stream itself (not just the raw
        # JSON dump below), so the sentiment result is visible right where
        # it completed, alongside every other step's log line.
        logger.info(
            "Result for request %s: status=%s sentiment=%s (score=%s) topic=%r error=%s",
            result.id,
            result.status,
            result.sentiment.label if result.sentiment else None,
            result.sentiment.score if result.sentiment else None,
            result.context.topic if result.context else None,
            result.error,
        )
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))

        # Mirror main.py's run_once() dev-dump behavior (spec revision-1
        # section 6) so a replay run is also useful for verifying
        # DEV_DUMP_OUTPUT end to end -- this script never publishes to
        # Kafka, so without this the dump would be the only way to inspect
        # a replayed result from disk afterward.
        if settings.dev_dump_output:
            _dump_result_for_dev(result, "output")


if __name__ == "__main__":
    main()
