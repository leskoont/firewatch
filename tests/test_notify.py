import logging

import httpx

from firewatch.cli import setup_logging
from firewatch.notify import TelegramNotifier

JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"


def make(handler, sleeps=None, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    n = TelegramNotifier("123:TOKEN", "-100", client=client, **kw)
    n._sleep = (sleeps if sleeps is not None else []).append
    return n


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


def test_client_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"ok": False, "description": "chat not found"})

    assert not make(handler, retries=3).send_text("x")
    assert len(calls) == 1


def test_server_error_is_retried():
    responses = iter([httpx.Response(502), httpx.Response(200, json={"ok": True})])
    assert make(lambda r: next(responses), retries=2).send_text("x")


def test_network_error():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    assert not make(handler, retries=1).send_text("x")


def test_rate_limit_waits_retry_after():
    responses = iter(
        [
            httpx.Response(429, json={"ok": False, "parameters": {"retry_after": 30}}),
            httpx.Response(429, json={"ok": False, "parameters": {"retry_after": 3600}}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    sleeps = []
    assert make(lambda r: next(responses), sleeps, retries=3).send_text("x")
    assert sleeps == [30, 60]


def test_non_json_reply_is_a_failure(caplog):
    n = make(lambda r: httpx.Response(200, text="<html>502 Bad Gateway</html>"), retries=1)
    assert not n.send_text("x")
    assert "giving up" in caplog.text


def test_token_is_not_in_errors(caplog):
    def handler(request):
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    assert not make(handler, retries=0).send_photo(JPEG)
    assert "cannot reach" in caplog.text
    assert "TOKEN" not in caplog.text


def test_setup_logging_quiets_httpx():
    setup_logging()
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)


def test_long_caption_is_trimmed():
    bodies = []

    def handler(request):
        bodies.append(request.read())
        return httpx.Response(200, json={"ok": True})

    make(handler).send_photo(JPEG, caption="a" * 5000)
    assert b"a" * 1024 in bodies[0] and b"a" * 1025 not in bodies[0]
