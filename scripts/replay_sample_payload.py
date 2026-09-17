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
from main import _dump_result_for_dev, build_dependencies  # noqa: E402
from messaging.scrapper_dto import NormalizedData, requests_from_normalized_data  # noqa: E402
from pipeline.analyze import analyze  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Real message captured from the scrapper-to-analysis topic (docs/to-do.md,
# 2026-09-17) -- includes one nested comment, so a replay run also exercises
# the per-comment analysis path alongside the post itself.
SAMPLE_PAYLOAD = {
    "id": "7685758857540275463",
    "platform": "tiktok",
    "message": "KPK Tangkap 17 Orang Termasuk Dirjen ATR/BPN #ott #korupsi #kpk",
    "url": "https://www.tiktok.com/@kompas.tv.ambon/video/7685758857540275463",
    "videoUrl": (
        "https://v19.tiktokcdn-us.com/085fdfbadbdd79956ba5178c1b555120/6aaba853/video/tos/"
        "alisg/tos-alisg-pve-0037c001/oEyT7AfBUqEQTTR2Epg4IeFEsqKFwqpVDUBUBU/"
        "?a=1233&bti=NEBzNTY6QGo6OjZALnAjNDQuYCMxNDNg&&bt=198"
        "&ft=arR-IqgmmklPD12cbW-I3wURSa3qjeF~O5&mime_type=video_mp4"
        "&rc=ZDRmaWg7OWk5OmRpOGYzNUBpajhrcnc5cmRuZDMzODczNEAtNTViNF5hNTIxMV8yYy8xYSM2cDRoMmRj"
        "bWhhLS1kMTFzcw%3D%3D&vvpl=1&l=20260917023639D137C72DF5E6A13EE9BC&btag=e000a0000"
    ),
    "imageUrl": None,
    "authorUsername": "kompas.tv.ambon",
    "authorName": "Kompas tv Ambon",
    "views": 47196,
    "likes": 1377,
    "repliesCount": 128,
    "uploadedAt": "2026-09-15T13:49:53.000Z",
    "comments": [
        {
            "id": "7685801363246236437",
            "message": "10+5=17",
            "url": None,
            "authorUsername": "sayfuladam7",
            "authorName": None,
            "likes": 1,
            "repliesCount": 0,
            "uploadedAt": "2026-09-15T16:34:47.000Z",
            "commentTo": "7685758857540275463",
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
