from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger(__name__)

API_URL = "https://api.telegram.org"
MAX_RETRY_AFTER_S = 60.0


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        retries: int = 2,
    ):
        self.chat_id = chat_id
        self.retries = retries
        self._base = f"{API_URL}/bot{token}"
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = time.sleep

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
        delay = 1.0
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(delay)
                delay *= 2
            try:
                resp = self._client.post(f"{self._base}/{method}", data=data, files=files)
            except httpx.HTTPError as e:
                error = f"{type(e).__name__}: {e}"
                log.warning("telegram %s failed (attempt %d): %s", method, attempt + 1, error)
                continue

            body = _json(resp)
            if resp.status_code == 200 and body.get("ok"):
                return True
            error = f"HTTP {resp.status_code} {body.get('description') or resp.text[:200]!r}"
            log.warning("telegram %s failed (attempt %d): %s", method, attempt + 1, error)
            if resp.status_code == 429:
                retry_after = (body.get("parameters") or {}).get("retry_after", delay)
                delay = min(float(retry_after), MAX_RETRY_AFTER_S)
            elif resp.status_code < 500:
                break
        log.error("telegram %s: giving up, %s", method, error)
        return False

    def close(self) -> None:
        self._client.close()


def _json(resp: httpx.Response) -> dict:
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
