import json

from messaging.scrapper_dto import NormalizedData, request_from_normalized_data
from schemas import Platform


def test_normalized_data_parses_camel_case_json_from_scrapper_be() -> None:
    data = NormalizedData(
        **{
            "id": "item-1",
            "platform": "twitter",
            "type": "POST",
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
            "commentTo": None,
        }
    )

    assert data.id == "item-1"
    assert data.platform == "twitter"
    assert data.type == "POST"
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
    assert data.comment_to is None


def test_normalized_data_allows_missing_optional_fields() -> None:
    data = NormalizedData(**{"id": "item-2"})

    assert data.video_url is None
    assert data.platform is None
    assert data.type is None


def test_request_from_normalized_data_maps_message_to_text_and_video_url() -> None:
    data = NormalizedData(
        **{
            "id": "item-1",
            "platform": "tiktok",
            "type": "POST",
            "message": "Judul post",
            "videoUrl": "https://cdn.example.com/video1.mp4",
        }
    )

    request = request_from_normalized_data(data)

    assert request.id == "item-1"
    assert request.text == "Judul post"
    assert request.video_url == "https://cdn.example.com/video1.mp4"
    assert request.platform == Platform.TIKTOK


def test_request_from_normalized_data_normalizes_facebook_casing() -> None:
    # Scrapper Backend's own Platform enum inconsistently serializes Facebook
    # as "Facebook" (capitalized) while every other platform is lowercase.
    data = NormalizedData(**{"id": "item-1", "platform": "Facebook", "message": "x"})

    request = request_from_normalized_data(data)

    assert request.platform == Platform.FACEBOOK


def test_request_from_normalized_data_returns_none_platform_for_unrecognized_value() -> None:
    data = NormalizedData(**{"id": "item-1", "platform": "myspace", "message": "x"})

    request = request_from_normalized_data(data)

    assert request.platform is None


def test_request_from_normalized_data_returns_none_platform_when_absent() -> None:
    data = NormalizedData(**{"id": "item-1", "message": "x"})

    request = request_from_normalized_data(data)

    assert request.platform is None


def test_parses_real_payload_captured_from_scrapper_be() -> None:
    # Exact message body pasted from a real scrapper-to-analysis Kafka
    # message (docs/to-do.md) -- locks in the real-world shape, including
    # which fields the Scrapper Backend actually sends as null.
    message_text = (
        "Harmoni Kecerdasan Pikiran dan Hati Pemimpin, "
        "Akan Melahirkan Piranti Kebijakan Original"
    )
    raw = json.dumps(
        {
            "id": "431559555550806016",
            "platform": "twitter",
            "type": "POST",
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
            "commentTo": None,
        }
    )

    data = NormalizedData(**json.loads(raw))
    request = request_from_normalized_data(data)

    assert request.id == "431559555550806016"
    assert request.platform == Platform.TWITTER
    assert request.text == message_text
    assert request.video_url is None
    assert request.metadata["author_username"] == "DediMulyadi71"
    assert request.metadata["views"] is None
    assert request.metadata["likes"] == 2
    assert request.metadata["uploaded_at"] == 1391726790


def test_request_from_normalized_data_carries_extra_fields_into_metadata() -> None:
    data = NormalizedData(
        **{
            "id": "item-1",
            "type": "COMMENT",
            "message": "Komentar",
            "commentTo": "item-0",
            "authorUsername": "commenter",
        }
    )

    request = request_from_normalized_data(data)

    assert request.metadata["type"] == "COMMENT"
    assert request.metadata["comment_to"] == "item-0"
    assert request.metadata["author_username"] == "commenter"
