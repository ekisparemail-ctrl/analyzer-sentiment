from enum import StrEnum

from pydantic import BaseModel, Field


class Platform(StrEnum):
    TWITTER = "twitter"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"


class Status(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


class AnalysisRequest(BaseModel):
    id: str
    # Optional: the Scrapper Backend's Kafka message (NormalizedDataDto) does
    # carry a platform field and it is populated from real Kafka messages
    # (see messaging/scrapper_dto.py) -- stays optional because it's still
    # not guaranteed/used for branching, and to keep this schema tolerant of
    # callers that don't know the platform.
    platform: Platform | None = None
    text: str | None = None
    video_url: str | None = None
    metadata: dict = Field(default_factory=dict)


class AnalysisContext(BaseModel):
    topic: str
    key_themes: list[str] = Field(default_factory=list)


class SentimentResult(BaseModel):
    label: str
    score: float
    indicators: list[str] = Field(default_factory=list)


class EmotionResult(BaseModel):
    primary: str
    secondary: str
    intensity: float
    indicators: list[str] = Field(default_factory=list)


class MotivationResult(BaseModel):
    type: str
    confidence_score: float
    indicators: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    id: str
    status: Status
    video_summary: str | None = None
    context: AnalysisContext | None = None
    sentiment: SentimentResult | None = None
    emotion: EmotionResult | None = None
    motivation: MotivationResult | None = None
    error: str | None = None
