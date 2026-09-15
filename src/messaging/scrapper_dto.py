from pydantic import BaseModel, ConfigDict, Field

from schemas import AnalysisRequest


class NormalizedPost(BaseModel):
    """
    Mirrors com.agora.dto.scrapping.NormalizedPostDto from the Scrapper Backend
    (Quarkus/Java), published as-is (Jackson field names) to the
    post-scrapper-to-analysis Kafka topic.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str | None = None
    post_url: str | None = Field(default=None, alias="postUrl")
    video_url: str | None = Field(default=None, alias="videoUrl")
    channel_username: str | None = Field(default=None, alias="channelUsername")
    channel_name: str | None = Field(default=None, alias="channelName")
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    uploaded_at: int | None = Field(default=None, alias="uploadedAt")


class NormalizedComment(BaseModel):
    """
    Mirrors com.agora.dto.scrapping.NormalizedCommentDto from the Scrapper Backend,
    published as-is to the comment-scrapper-to-analysis Kafka topic.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    post_id: str | None = Field(default=None, alias="postId")
    text: str | None = None
    username: str | None = None
    likes: int | None = None
    replies: int | None = None
    created_at: str | None = Field(default=None, alias="createdAt")
    has_media: bool | None = Field(default=None, alias="hasMedia")


def request_from_post(post: NormalizedPost) -> AnalysisRequest:
    """
    Neither NormalizedPostDto nor NormalizedCommentDto carries a platform field,
    so AnalysisRequest.platform is left unset (None) here -- there is currently no
    source data to populate it from.
    """
    return AnalysisRequest(
        id=post.id,
        text=post.title,
        video_url=post.video_url,
        metadata={
            "source": "post",
            "post_url": post.post_url,
            "channel_username": post.channel_username,
            "channel_name": post.channel_name,
            "views": post.views,
            "likes": post.likes,
            "comments": post.comments,
            "uploaded_at": post.uploaded_at,
        },
    )


def request_from_comment(comment: NormalizedComment) -> AnalysisRequest:
    return AnalysisRequest(
        id=comment.id,
        text=comment.text,
        video_url=None,
        metadata={
            "source": "comment",
            "post_id": comment.post_id,
            "username": comment.username,
            "likes": comment.likes,
            "replies": comment.replies,
            "created_at": comment.created_at,
            "has_media": comment.has_media,
        },
    )
