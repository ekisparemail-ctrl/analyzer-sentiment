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

from config import Settings  # noqa: E402
from main import build_dependencies  # noqa: E402
from messaging.scrapper_dto import NormalizedData, requests_from_normalized_data  # noqa: E402
from pipeline.analyze import analyze  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Real message captured from the scrapper-to-analysis topic (docs/to-do.md,
# 2026-09-16 07:09:15.589, partition 0 offset 12) -- a shorter (~1 min)
# video, swapped in to keep replay runs quick during iteration.
SAMPLE_PAYLOAD = {
    "id": "7684242422607482133",
    "platform": "tiktok",
    "message": (
        "Febrie Pegang Kartu Truf! Boyamin Pernah Dikasih 100.000 SGD?  "
        "#febrieadriansyah #korupsi #jampidsus #boyamin #uang "
    ),
    "url": "https://www.tiktok.com/@opinijend/video/7684242422607482133",
    "videoUrl": (
        "https://v19.tiktokcdn-us.com/340d670c838c0a55317bca04194193ff/6aaa953c/video/tos/"
        "alisg/tos-alisg-pve-0037c001/oc4JRs4jLQs8I64DEVeAKmDoFVfgACdKeIwjlQ/"
        "?a=1233&bti=NEBzNTY6QGo6OjZALnAjNDQuYCMxNDNg&&bt=567"
        "&ft=WgSBMNTYVUywUytrzLnq2Ef5SxYnD1PXtFksencyqF_4&mime_type=video_mp4"
        "&rc=ODdkNWhmaGZmM2llNjQ0NEBpMzhyOm45cjh4ZDMzODczNEA1XzJiMV4vNi0xNF8vX2FfYSNvNjBfMmRj"
        "Z2VhLS1kMWBzcw%3D%3D&vvpl=1&l=202609160709039A2F2E99CE5DB00FF157&btag=e000d0000"
    ),
    "imageUrl": None,
    "authorUsername": "opinijend",
    "authorName": "OpiniJend",
    "views": 286252,
    "likes": 5525,
    "repliesCount": 292,
    "uploadedAt": "2026-09-11T11:45:14.000Z",
    "comments": [],
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


if __name__ == "__main__":
    main()
