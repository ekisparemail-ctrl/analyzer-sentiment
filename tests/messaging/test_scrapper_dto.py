from messaging.scrapper_dto import (
    NormalizedComment,
    NormalizedPost,
    request_from_comment,
    request_from_post,
)


def test_normalized_post_parses_camel_case_json_from_scrapper_be() -> None:
    post = NormalizedPost(
        **{
            "id": "post-1",
            "title": "Contoh judul post",
            "postUrl": "https://tiktok.com/@user/video/1",
            "videoUrl": "https://cdn.example.com/video1.mp4",
            "channelUsername": "user",
            "channelName": "User Name",
            "views": 100,
            "likes": 10,
            "comments": 2,
            "uploadedAt": 1700000000,
        }
    )

    assert post.id == "post-1"
    assert post.title == "Contoh judul post"
    assert post.post_url == "https://tiktok.com/@user/video/1"
    assert post.video_url == "https://cdn.example.com/video1.mp4"
    assert post.channel_username == "user"
    assert post.channel_name == "User Name"
    assert post.views == 100
    assert post.likes == 10
    assert post.comments == 2
    assert post.uploaded_at == 1700000000


def test_normalized_post_allows_missing_video_url() -> None:
    post = NormalizedPost(**{"id": "post-2", "title": "Tanpa video"})

    assert post.video_url is None


def test_normalized_comment_parses_camel_case_json_from_scrapper_be() -> None:
    comment = NormalizedComment(
        **{
            "id": "comment-1",
            "postId": "post-1",
            "text": "Komentar contoh",
            "username": "commenter",
            "likes": 3,
            "replies": 1,
            "createdAt": "2025-01-01T00:00:00Z",
            "hasMedia": False,
        }
    )

    assert comment.id == "comment-1"
    assert comment.post_id == "post-1"
    assert comment.text == "Komentar contoh"
    assert comment.username == "commenter"
    assert comment.likes == 3
    assert comment.replies == 1
    assert comment.created_at == "2025-01-01T00:00:00Z"
    assert comment.has_media is False


def test_request_from_post_uses_title_as_text_and_carries_video_url() -> None:
    post = NormalizedPost(
        **{
            "id": "post-1",
            "title": "Judul post",
            "videoUrl": "https://cdn.example.com/video1.mp4",
            "channelUsername": "user",
        }
    )

    request = request_from_post(post)

    assert request.id == "post-1"
    assert request.text == "Judul post"
    assert request.video_url == "https://cdn.example.com/video1.mp4"
    assert request.platform is None
    assert request.metadata["channel_username"] == "user"


def test_request_from_post_without_video_url_leaves_video_url_none() -> None:
    post = NormalizedPost(**{"id": "post-2", "title": "Tanpa video"})

    request = request_from_post(post)

    assert request.video_url is None


def test_request_from_comment_uses_text_and_never_has_video_url() -> None:
    comment = NormalizedComment(
        **{"id": "comment-1", "postId": "post-1", "text": "Komentar contoh"}
    )

    request = request_from_comment(comment)

    assert request.id == "comment-1"
    assert request.text == "Komentar contoh"
    assert request.video_url is None
    assert request.platform is None
    assert request.metadata["post_id"] == "post-1"
