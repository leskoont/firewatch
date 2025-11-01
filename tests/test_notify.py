import httpx

from firewatch.notify import TelegramNotifier

JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"


def make(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TelegramNotifier("123:TOKEN", "-100", client=client)


def test_send_photo_posts_multipart():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "result": {}})

    n = make(handler)
    assert n.send_photo(JPEG, caption="fire on cam1")

    req = seen[0]
    assert req.url.path == "/bot123:TOKEN/sendPhoto"
    assert req.headers["content-type"].startswith("multipart/form-data")
    body = req.read()
    assert b'name="chat_id"' in body and b"-100" in body
    assert b"fire on cam1" in body
    assert b'filename="snapshot.jpg"' in body
    assert JPEG in body


def test_send_text():
    def handler(request):
        assert request.url.path.endswith("/sendMessage")
        return httpx.Response(200, json={"ok": True})

    assert make(handler).send_text("hello")


def test_api_error():
    def handler(request):
        return httpx.Response(400, json={"ok": False, "description": "chat not found"})

    assert not make(handler).send_text("x")


def test_network_error():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    assert not make(handler).send_text("x")


def test_non_json_reply_is_a_failure():
    assert not make(lambda r: httpx.Response(200, text="<html>502 Bad Gateway</html>")).send_text("x")


def test_long_caption_is_trimmed():
    bodies = []

    def handler(request):
        bodies.append(request.read())
        return httpx.Response(200, json={"ok": True})

    make(handler).send_photo(JPEG, caption="a" * 5000)
    assert b"a" * 1024 in bodies[0] and b"a" * 1025 not in bodies[0]
