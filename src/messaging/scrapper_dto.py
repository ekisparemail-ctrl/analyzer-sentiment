from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas import AnalysisRequest, Platform


def _normalize_uploaded_at(value: object) -> object:
    # The Scrapper Backend inconsistently serializes uploadedAt: an epoch
    # integer for some items, an ISO-8601 string ("...T...Z") for others
    # (observed for TikTok comments). This field is carried through only
    # for traceability (see request builders below), so normalize it to
    # epoch seconds -- or drop it to None if it's neither -- rather than
    # rejecting the entire message over one non-critical field.
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


class NormalizedComment(BaseModel):
    """A single comment nested under a NormalizedData post (see below)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    message: str | None = None
    url: str | None = None
    author_username: str | None = Field(default=None, alias="authorUsername")
    author_name: str | None = Field(default=None, alias="authorName")
    likes: int | None = None
    replies_count: int | None = Field(default=None, alias="repliesCount")
    uploaded_at: int | None = Field(default=None, alias="uploadedAt")
    comment_to: str | None = Field(default=None, alias="commentTo")

    @field_validator("uploaded_at", mode="before")
    @classmethod
    def _validate_uploaded_at(cls, value: object) -> object:
        return _normalize_uploaded_at(value)


class NormalizedData(BaseModel):
    """
    Mirrors com.agora.dto.scrapping.NormalizedDataDto from the Scrapper Backend
    (Quarkus/Java), published as-is (Jackson field names) to the single
    scrapper-to-analysis Kafka topic.

    Revised again (2026-09-17): the Scrapper Backend dropped the top-level
    `type`/`commentTo` fields and now nests a post's comments directly
    inside it as a `comments` array, instead of publishing each comment as
    its own separate top-level message. See requests_from_normalized_data().
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    platform: str | None = None
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
    comments: list[NormalizedComment] = Field(default_factory=list)

    @field_validator("uploaded_at", mode="before")
    @classmethod
    def _validate_uploaded_at(cls, value: object) -> object:
        return _normalize_uploaded_at(value)


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


def _post_request(data: NormalizedData, platform: Platform | None) -> AnalysisRequest:
    return AnalysisRequest(
        id=data.id,
        platform=platform,
        text=data.message,
        video_url=data.video_url,
        metadata={
            "type": "POST",
            "url": data.url,
            "image_url": data.image_url,
            "author_username": data.author_username,
            "author_name": data.author_name,
            "views": data.views,
            "likes": data.likes,
            "replies_count": data.replies_count,
            "uploaded_at": data.uploaded_at,
        },
    )


def _comment_request(comment: NormalizedComment, platform: Platform | None) -> AnalysisRequest:
    return AnalysisRequest(
        id=comment.id,
        platform=platform,
        text=comment.message,
        video_url=None,
        metadata={
            "type": "COMMENT",
            "url": comment.url,
            "author_username": comment.author_username,
            "author_name": comment.author_name,
            "likes": comment.likes,
            "replies_count": comment.replies_count,
            "uploaded_at": comment.uploaded_at,
            "comment_to": comment.comment_to,
        },
    )


def requests_from_normalized_data(data: NormalizedData) -> list[AnalysisRequest]:
    """
    One NormalizedData message now yields the post itself plus one request
    per nested comment -- the Scrapper Backend stopped publishing each
    comment as its own separate top-level message (see class docstring).
    Comments inherit the parent post's platform (they don't carry their own)
    and never have a video_url (only posts do).
    """
    platform = _parse_platform(data.platform)
    requests = [_post_request(data, platform)]
    requests.extend(_comment_request(comment, platform) for comment in data.comments)
    return requests
