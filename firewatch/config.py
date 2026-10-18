from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints, model_validator

from firewatch.events import safe_name
from firewatch.stream import userinfo_end

log = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _env(m: re.Match) -> str:
    name, default = m.groups()
    value = os.environ.get(name, "")
    if not value and default is None:
        log.warning("${%s} is not set, using an empty string", name)
    # like in the shell, ${VAR:-x} gives x for an empty VAR too
    return value or default or ""


def expand_env(value: Any) -> Any:
    """Substitute ${VAR} and ${VAR:-default} in all strings of a parsed YAML tree."""
    if isinstance(value, str):
        return _ENV_RE.sub(_env, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


class Strict(BaseModel):
    # typos in keys fail at startup
    model_config = ConfigDict(extra="forbid")


Score = Annotated[float, Field(ge=0, le=1)]


def _as_text(v: Any) -> Any:
    # `chat_id:` with no value comes from YAML as None, a numeric chat_id or webcam index as int
    if v is None:
        return ""
    return str(v) if isinstance(v, int) and not isinstance(v, bool) else v


Text = Annotated[str, BeforeValidator(_as_text), StringConstraints(strip_whitespace=True)]
NonBlank = Annotated[Text, StringConstraints(min_length=1)]


class ModelConfig(Strict):
    path: Path = Path("models/firewatch-yolo11s.onnx")
    providers: str | list[str] = "auto"
    conf: Score = 0.25
    iou: Score = 0.45
    threads: int | None = Field(None, ge=1)


class TelegramConfig(Strict):
    enabled: bool = False
    token: Text = ""
    chat_id: Text = ""
    timeout: float = Field(10.0, gt=0)

    @model_validator(mode="after")
    def _check_credentials(self):
        if self.enabled and not (self.token and self.chat_id):
            raise ValueError("telegram is enabled but token or chat_id is empty")
        return self


class StorageConfig(Strict):
    dir: Path = Path("events")
    snapshots: bool = True
    clips: bool = False
    clip_before_s: float = Field(3.0, ge=0)
    clip_after_s: float = Field(5.0, ge=0)


class CameraConfig(Strict):
    name: NonBlank
    source: NonBlank
    enabled: bool = True
    process_every: int = Field(3, ge=1)
    window: int = Field(10, ge=1)
    min_hits: int = Field(6, ge=1)
    cooldown_s: float = Field(300.0, ge=0)
    # per-class confidence; classes not listed here never raise an alert
    thresholds: dict[str, Score] = Field(default_factory=lambda: {"fire": 0.4, "smoke": 0.5}, min_length=1)
    reconnect_delay_s: float = Field(5.0, ge=0)
    loop: bool = False  # restart video files at EOF, handy for demos

    @model_validator(mode="after")
    def _check_window(self):
        if self.min_hits > self.window:
            raise ValueError(f"camera {self.name}: min_hits ({self.min_hits}) > window ({self.window})")
        return self

    @property
    def is_live(self) -> bool:
        return self.source.isdigit() or "://" in self.source


class ApiConfig(Strict):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


class Settings(Strict):
    model: ModelConfig = ModelConfig()
    telegram: TelegramConfig = TelegramConfig()
    storage: StorageConfig = StorageConfig()
    api: ApiConfig = ApiConfig()
    cameras: list[CameraConfig] = []

    @model_validator(mode="after")
    def _unique_names(self):
        names = [c.name for c in self.cameras]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate camera names: {sorted(dupes)}")
        # event files are named after the camera, "Склад 1" and "склад/1" would share them
        by_file: dict[str, list[str]] = {}
        for n in names:
            by_file.setdefault(safe_name(n).casefold(), []).append(n)
        clashes = [group for group in by_file.values() if len(group) > 1]
        if clashes:
            raise ValueError(f"camera names that map to the same event file name: {clashes}")
        return self

    @model_validator(mode="after")
    def _warn_unencoded_password(self):
        for cam in self.cameras:
            if _raw_reserved_in_password(cam.source):
                log.warning(
                    "camera %s: the password in source contains a raw '#', '?' or '/', "
                    "encode them as %%23, %%3F and %%2F or the stream won't open",
                    cam.name,
                )
        return self

    @model_validator(mode="after")
    def _warn_low_thresholds(self):
        for cam in self.cameras:
            low = sorted(k for k, v in cam.thresholds.items() if v < self.model.conf)
            if low:
                log.warning(
                    "camera %s: thresholds for %s are below model.conf=%g and have no effect",
                    cam.name,
                    low,
                    self.model.conf,
                )
        return self

    def check_classes(self, names: Iterable[str]) -> None:
        names = set(names)
        for cam in self.cameras:
            unknown = set(cam.thresholds) - names
            if unknown:
                raise ValueError(
                    f"camera {cam.name}: thresholds for {sorted(unknown)}, "
                    f"but the model only knows {sorted(names)}"
                )


def _raw_reserved_in_password(source: str) -> bool:
    # FFmpeg ends the host part at the first '/', '?' or '#', so none of them may stay
    # unencoded in user:password
    end = userinfo_end(source)
    return end >= 0 and bool(re.search(r"[/?#]", source[source.find("://") + 3 : end]))


def _merge_camera(defaults: dict, cam: Any) -> dict:
    if not isinstance(cam, dict):
        raise ValueError(f"every entry in cameras must be a mapping, got {type(cam).__name__}")
    # `thresholds:` left empty means "the defaults", not "no classes"
    cam = {k: v for k, v in cam.items() if not (k == "thresholds" and v is None)}
    merged = {**defaults, **cam}
    if isinstance(defaults.get("thresholds"), dict) and isinstance(cam.get("thresholds"), dict):
        merged["thresholds"] = {**defaults["thresholds"], **cam["thresholds"]}
    if isinstance(merged.get("thresholds"), dict):
        merged["thresholds"] = {k: v for k, v in merged["thresholds"].items() if v is not None}
    return merged


def load_settings(path: str | Path) -> Settings:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("the config must be a YAML mapping")

    defaults = raw.pop("camera_defaults", None) or {}
    cameras = raw.get("cameras") or []
    if not isinstance(defaults, dict) or not isinstance(cameras, list):
        raise ValueError("camera_defaults must be a mapping and cameras a list")
    raw["cameras"] = [_merge_camera(defaults, cam) for cam in cameras]
    return Settings.model_validate(expand_env(raw))
