from mcp_prism.gateway import last_user_text


def test_last_user_text_supports_multimodal_shape():
    messages = [{"role": "user", "content": [{"type": "text", "text": "서울 날씨"}, {"type": "image_url"}]}]
    assert last_user_text(messages) == "서울 날씨"

