from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

API_URL = "https://api.telegram.org"


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ):
        self.chat_id = chat_id
        self._base = f"{API_URL}/bot{token}"
        self._client = client or httpx.Client(timeout=timeout)

    def send_text(self, text: str) -> bool:
        return self._post("sendMessage", data={"chat_id": self.chat_id, "text": text})

    def send_photo(self, jpeg: bytes, caption: str = "") -> bool:
        if not jpeg:
            return False
        return self._post(
            "sendPhoto",
            data={"chat_id": self.chat_id, "caption": caption[:1024]},
            files={"photo": ("snapshot.jpg", jpeg, "image/jpeg")},
        )

    def _post(self, method: str, data: dict, files: dict | None = None) -> bool:
        try:
            resp = self._client.post(f"{self._base}/{method}", data=data, files=files)
        except httpx.HTTPError as e:
            log.error("telegram %s failed: %s", method, e)
            return False
        body = _json(resp)
        if resp.status_code == 200 and body.get("ok"):
            return True
        error = body.get("description") or resp.text[:200]
        log.error("telegram %s failed: HTTP %s %r", method, resp.status_code, error)
        return False

    def close(self) -> None:
        self._client.close()


def _json(resp: httpx.Response) -> dict:
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
