from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas import AnalysisRequest, Platform


class NormalizedData(BaseModel):
    """
    Mirrors com.agora.dto.scrapping.NormalizedDataDto from the Scrapper Backend
    (Quarkus/Java), published as-is (Jackson field names) to the single
    scrapper-to-analysis Kafka topic. Replaces the earlier separate
    NormalizedPost/NormalizedComment DTOs after the Scrapper Backend merged
    its post/comment/reply producers into one.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    platform: str | None = None
    type: str | None = None  # "POST" | "COMMENT" | "REPLY"
    message: str | None = None
    url: str | None = None
    video_url: str | None = Field(default=None, alias="videoUrl")
    image_url: str | None = Field(default=None, alias="imageUrl")
    author_username: str | None = Field(default=None, alias="authorUsername")
    author_name: str | None = Field(default=None, alias="authorName")
    views: int | None = None
    likes: int | None = None
    replies_count: int | None = Field(default=None, alias="repliesCount")
    uploaded_at: int | None = Field(default=None, alias="uploadedAt")
    comment_to: str | None = Field(default=None, alias="commentTo")

    @field_validator("uploaded_at", mode="before")
    @classmethod
    def _normalize_uploaded_at(cls, value: object) -> object:
        # The Scrapper Backend inconsistently serializes uploadedAt: an epoch
        # integer for some items, an ISO-8601 string ("...T...Z") for others
        # (observed for TikTok comments). This field is carried through only
        # for traceability (see request_from_normalized_data), so normalize
        # it to epoch seconds -- or drop it to None if it's neither -- rather
        # than rejecting the entire message over one non-critical field.
        if not isinstance(value, str):
            return value
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return None


def _parse_platform(raw: str | None) -> Platform | None:
    if raw is None:
        return None
    try:
        # The Scrapper Backend's own Platform enum inconsistently serializes
        # Facebook as "Facebook" (capitalized) while every other platform is
        # lowercase -- normalize case rather than fail validation over it.
        return Platform(raw.lower())
    except ValueError:
        return None


def request_from_normalized_data(data: NormalizedData) -> AnalysisRequest:
    return AnalysisRequest(
        id=data.id,
        platform=_parse_platform(data.platform),
        text=data.message,
        video_url=data.video_url,
        metadata={
            "type": data.type,
            "url": data.url,
            "image_url": data.image_url,
            "author_username": data.author_username,
            "author_name": data.author_name,
            "views": data.views,
            "likes": data.likes,
            "replies_count": data.replies_count,
            "uploaded_at": data.uploaded_at,
            "comment_to": data.comment_to,
        },
    )
