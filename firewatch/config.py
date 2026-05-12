from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

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


Score = Annotated[float, Field(ge=0, le=1)]


class ModelConfig(BaseModel):
    path: Path = Path("models/firewatch-yolo11s.onnx")
    providers: str | list[str] = "auto"
    conf: Score = 0.25
    iou: Score = 0.45
    threads: int | None = Field(None, ge=1)


class TelegramConfig(BaseModel):
    enabled: bool = False
    token: str = ""
    chat_id: str = ""
    timeout: float = Field(10.0, gt=0)

    @field_validator("chat_id", mode="before")
    @classmethod
    def _chat_id_as_str(cls, v):
        # `chat_id:` with no value comes from YAML as None
        return "" if v is None else str(v)

    @model_validator(mode="after")
    def _check_credentials(self):
        if self.enabled and not (self.token and self.chat_id):
            raise ValueError("telegram is enabled but token or chat_id is empty")
        return self


class StorageConfig(BaseModel):
    dir: Path = Path("events")
    snapshots: bool = True


class CameraConfig(BaseModel):
    name: str
    source: str
    enabled: bool = True
    process_every: int = Field(3, ge=1)
    window: int = Field(10, ge=1)
    min_hits: int = Field(6, ge=1)
    cooldown_s: float = Field(300.0, ge=0)
    # per-class confidence; classes not listed here never raise an alert
    thresholds: dict[str, Score] = Field(default_factory=lambda: {"fire": 0.4, "smoke": 0.5})
    reconnect_delay_s: float = Field(5.0, ge=0)
    loop: bool = False  # restart video files at EOF, handy for demos

    @field_validator("source", mode="before")
    @classmethod
    def _source_as_str(cls, v):
        return str(v) if isinstance(v, int) else v

    @model_validator(mode="after")
    def _check_window(self):
        if self.min_hits > self.window:
            raise ValueError(f"camera {self.name}: min_hits ({self.min_hits}) > window ({self.window})")
        return self

    @property
    def is_live(self) -> bool:
        return self.source.isdigit() or "://" in self.source


class ApiConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


class Settings(BaseModel):
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
        return self


def _merge_camera(defaults: dict, cam: dict) -> dict:
    merged = {**defaults, **cam}
    if isinstance(defaults.get("thresholds"), dict) and isinstance(cam.get("thresholds"), dict):
        merged["thresholds"] = {**defaults["thresholds"], **cam["thresholds"]}
    if isinstance(merged.get("thresholds"), dict):
        merged["thresholds"] = {k: v for k, v in merged["thresholds"].items() if v is not None}
    return merged


def load_settings(path: str | Path) -> Settings:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.pop("camera_defaults", None) or {}
    raw["cameras"] = [_merge_camera(defaults, cam) for cam in raw.get("cameras") or []]
    return Settings.model_validate(expand_env(raw))
