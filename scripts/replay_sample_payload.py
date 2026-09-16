"""
Dev-only tool: runs the full Analytics Backend pipeline (download, VLM,
whisper, LLM) against a hardcoded Scrapper Backend payload, bypassing Kafka
entirely.

Use this when the Scrapper Backend can't produce live traffic (e.g. its own
AI credits run out again) but the pipeline still needs to be exercised
end-to-end against a real message. Swap SAMPLE_PAYLOAD for whatever message
you need to replay -- paste the exact JSON as seen in Kafdrop/Kafka, camelCase
and all (it goes through the same NormalizedData parsing real Kafka messages
do).

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
from messaging.scrapper_dto import NormalizedData, request_from_normalized_data  # noqa: E402
from pipeline.analyze import analyze  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Real message captured from the scrapper-to-analysis topic (docs/to-do.md,
# 2026-09-16, partition 0 offset 73).
SAMPLE_PAYLOAD = {
    "id": "7685758857540275463",
    "platform": "tiktok",
    "type": "POST",
    "message": "KPK Tangkap 17 Orang Termasuk Dirjen ATR/BPN #ott #korupsi #kpk",
    "url": "https://www.tiktok.com/@kompas.tv.ambon/video/7685758857540275463",
    "videoUrl": (
        "https://v16m.tiktokcdn-us.com/d0c29d97db51a6db354a2fe27a857c87/6aaaab1a/video/tos/"
        "alisg/tos-alisg-pve-0037c001/oEyT7AfBUqEQTTR2Epg4IeFEsqKFwqpVDUBUBU/"
        "?a=1233&bti=NEBzNTY6QGo6OjZALnAjNDQuYCMxNDNg&&bt=198"
        "&ft=arR-Iq4fmr2PD12lJU-I3wUEI7JUMeF~O5&mime_type=video_mp4"
        "&rc=ZDRmaWg7OWk5OmRpOGYzNUBpajhrcnc5cmRuZDMzODczNEAtNTViNF5hNTIxMV8yYy8xYSM2cDRoMmRj"
        "bWhhLS1kMTFzcw%3D%3D&vvpl=1&l=2026091608361478C3BF2E06E59D10F481&btag=e000d0000"
    ),
    "imageUrl": None,
    "authorUsername": "kompas.tv.ambon",
    "authorName": "Kompas tv Ambon",
    "views": 39291,
    "likes": 1036,
    "repliesCount": 91,
    "uploadedAt": "2026-09-15T13:49:53.000Z",
    "commentTo": None,
}


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    deps = build_dependencies(settings)

    request = request_from_normalized_data(NormalizedData(**SAMPLE_PAYLOAD))
    result = analyze(request, deps)

    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
