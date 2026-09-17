import json

from messaging.scrapper_dto import NormalizedComment, NormalizedData, requests_from_normalized_data
from schemas import Platform


def test_normalized_data_parses_camel_case_json_from_scrapper_be() -> None:
    data = NormalizedData(
        **{
            "id": "item-1",
            "platform": "twitter",
            "message": "Contoh teks postingan",
            "url": "https://twitter.com/user/status/1",
            "videoUrl": "https://cdn.example.com/video1.mp4",
            "imageUrl": "https://cdn.example.com/image1.jpg",
            "authorUsername": "user",
            "authorName": "User Name",
            "views": 100,
            "likes": 10,
            "repliesCount": 2,
            "uploadedAt": 1700000000,
        }
    )

    assert data.id == "item-1"
    assert data.platform == "twitter"
    assert data.message == "Contoh teks postingan"
    assert data.url == "https://twitter.com/user/status/1"
    assert data.video_url == "https://cdn.example.com/video1.mp4"
    assert data.image_url == "https://cdn.example.com/image1.jpg"
    assert data.author_username == "user"
    assert data.author_name == "User Name"
    assert data.views == 100
    assert data.likes == 10
    assert data.replies_count == 2
    assert data.uploaded_at == 1700000000
    assert data.comments == []


def test_normalized_data_allows_missing_optional_fields() -> None:
    data = NormalizedData(**{"id": "item-2"})

    assert data.video_url is None
    assert data.platform is None
    assert data.comments == []


def test_normalized_data_parses_nested_comments() -> None:
    data = NormalizedData(
        **{
            "id": "post-1",
            "platform": "tiktok",
            "message": "Judul post",
            "comments": [
                {
                    "id": "comment-1",
                    "message": "komentar pertama",
                    "url": None,
                    "authorUsername": "commenter1",
                    "authorName": None,
                    "likes": 1,
                    "repliesCount": 0,
                    "uploadedAt": 1700000100,
                    "commentTo": "post-1",
                }
            ],
        }
    )

    assert len(data.comments) == 1
    comment = data.comments[0]
    assert isinstance(comment, NormalizedComment)
    assert comment.id == "comment-1"
    assert comment.message == "komentar pertama"
    assert comment.author_username == "commenter1"
    assert comment.likes == 1
    assert comment.replies_count == 0
    assert comment.uploaded_at == 1700000100
    assert comment.comment_to == "post-1"


def test_normalized_comment_normalizes_iso_timestamp_uploaded_at() -> None:
    comment = NormalizedComment(
        **{"id": "c1", "message": "x", "uploadedAt": "2026-09-11T11:45:14.000Z"}
    )

    assert isinstance(comment.uploaded_at, int)


def test_requests_from_normalized_data_maps_message_to_text_and_video_url() -> None:
    data = NormalizedData(
        **{
            "id": "item-1",
            "platform": "tiktok",
            "message": "Judul post",
            "videoUrl": "https://cdn.example.com/video1.mp4",
        }
    )

    requests = requests_from_normalized_data(data)

    assert len(requests) == 1
    post_request = requests[0]
    assert post_request.id == "item-1"
    assert post_request.text == "Judul post"
    assert post_request.video_url == "https://cdn.example.com/video1.mp4"
    assert post_request.platform == Platform.TIKTOK
    assert post_request.metadata["type"] == "POST"


def test_requests_from_normalized_data_normalizes_facebook_casing() -> None:
    # Scrapper Backend's own Platform enum inconsistently serializes Facebook
    # as "Facebook" (capitalized) while every other platform is lowercase.
    data = NormalizedData(**{"id": "item-1", "platform": "Facebook", "message": "x"})

    requests = requests_from_normalized_data(data)

    assert requests[0].platform == Platform.FACEBOOK


def test_requests_from_normalized_data_returns_none_platform_for_unrecognized_value() -> None:
    data = NormalizedData(**{"id": "item-1", "platform": "myspace", "message": "x"})

    requests = requests_from_normalized_data(data)

    assert requests[0].platform is None


def test_requests_from_normalized_data_returns_none_platform_when_absent() -> None:
    data = NormalizedData(**{"id": "item-1", "message": "x"})

    requests = requests_from_normalized_data(data)

    assert requests[0].platform is None


def test_requests_from_normalized_data_carries_extra_fields_into_metadata() -> None:
    data = NormalizedData(
        **{
            "id": "item-1",
            "message": "Post asli",
            "authorUsername": "poster",
        }
    )

    requests = requests_from_normalized_data(data)

    assert requests[0].metadata["author_username"] == "poster"


def test_requests_from_normalized_data_yields_one_request_per_comment_plus_the_post() -> None:
    # Revised again: the Scrapper Backend stopped publishing each comment as
    # its own separate top-level message -- comments now arrive nested
    # inside their parent post's message, and one Kafka message can produce
    # several analyzable items.
    data = NormalizedData(
        **{
            "id": "post-1",
            "platform": "tiktok",
            "message": "Judul post",
            "videoUrl": "https://cdn.example.com/video1.mp4",
            "comments": [
                {"id": "comment-1", "message": "komentar 1", "commentTo": "post-1"},
                {"id": "comment-2", "message": "komentar 2", "commentTo": "post-1"},
            ],
        }
    )

    requests = requests_from_normalized_data(data)

    assert [r.id for r in requests] == ["post-1", "comment-1", "comment-2"]
    post_request, comment_1, comment_2 = requests
    assert post_request.video_url == "https://cdn.example.com/video1.mp4"
    assert post_request.metadata["type"] == "POST"
    # Comments never carry video, inherit the parent post's platform, and
    # are tagged with their own type + which post they belong to.
    assert comment_1.video_url is None
    assert comment_1.text == "komentar 1"
    assert comment_1.platform == Platform.TIKTOK
    assert comment_1.metadata["type"] == "COMMENT"
    assert comment_1.metadata["comment_to"] == "post-1"
    assert comment_2.id == "comment-2"


def test_requests_from_normalized_data_returns_only_the_post_when_no_comments() -> None:
    data = NormalizedData(**{"id": "post-1", "message": "Judul post"})

    requests = requests_from_normalized_data(data)

    assert len(requests) == 1
    assert requests[0].id == "post-1"


def test_parses_real_payload_captured_from_scrapper_be() -> None:
    # Exact message body pasted from a real scrapper-to-analysis Kafka
    # message (docs/to-do.md) -- locks in the real-world shape, including
    # the nested comments array introduced when the Scrapper Backend
    # stopped publishing each comment as its own top-level message.
    message_text = (
        "Harmoni Kecerdasan Pikiran dan Hati Pemimpin, "
        "Akan Melahirkan Piranti Kebijakan Original"
    )
    raw = json.dumps(
        {
            "id": "431559555550806016",
            "platform": "twitter",
            "message": message_text,
            "url": "https://x.com/DediMulyadi71/status/431559555550806016",
            "videoUrl": None,
            "imageUrl": None,
            "authorUsername": "DediMulyadi71",
            "authorName": "Kang Dedi Mulyadi",
            "views": None,
            "likes": 2,
            "repliesCount": 1,
            "uploadedAt": 1391726790,
            "comments": [],
        }
    )

    data = NormalizedData(**json.loads(raw))
    requests = requests_from_normalized_data(data)

    assert len(requests) == 1
    request = requests[0]
    assert request.id == "431559555550806016"
    assert request.platform == Platform.TWITTER
    assert request.text == message_text
    assert request.video_url is None
    assert request.metadata["author_username"] == "DediMulyadi71"
    assert request.metadata["views"] is None
    assert request.metadata["likes"] == 2
    assert request.metadata["uploaded_at"] == 1391726790


def test_parses_real_payload_with_nested_comment_captured_from_scrapper_be() -> None:
    # Exact message body pasted from docs/to-do.md (2026-09-17) after the
    # Scrapper Backend dropped the top-level type/commentTo fields and
    # started nesting a post's comments inside it.
    raw = json.dumps(
        {
            "id": "7685758857540275463",
            "platform": "tiktok",
            "message": "KPK Tangkap 17 Orang Termasuk Dirjen ATR/BPN #ott #korupsi #kpk",
            "url": "https://www.tiktok.com/@kompas.tv.ambon/video/7685758857540275463",
            "videoUrl": (
                "https://v19.tiktokcdn-us.com/085fdfbadbdd79956ba5178c1b555120/6aaba853/"
                "video/tos/alisg/tos-alisg-pve-0037c001/oEyT7AfBUqEQTTR2Epg4IeFEsqKFwqpVDUBUBU/"
                "?a=1233&bti=NEBzNTY6QGo6OjZALnAjNDQuYCMxNDNg&&bt=198"
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
    )

    data = NormalizedData(**json.loads(raw))
    requests = requests_from_normalized_data(data)

    assert len(requests) == 2
    post_request, comment_request = requests
    assert post_request.id == "7685758857540275463"
    assert post_request.platform == Platform.TIKTOK
    assert post_request.video_url is not None
    assert comment_request.id == "7685801363246236437"
    assert comment_request.text == "10+5=17"
    assert comment_request.platform == Platform.TIKTOK
    assert comment_request.video_url is None
    assert comment_request.metadata["comment_to"] == "7685758857540275463"
