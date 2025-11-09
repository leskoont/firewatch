from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

Score = Annotated[float, Field(ge=0, le=1)]


class ModelConfig(BaseModel):
    path: Path = Path("models/firewatch-yolo11s.onnx")
    providers: str | list[str] = "auto"
    conf: Score = 0.25
    iou: Score = 0.45


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


class Settings(BaseModel):
    model: ModelConfig = ModelConfig()
    telegram: TelegramConfig = TelegramConfig()
    storage: StorageConfig = StorageConfig()
    cameras: list[CameraConfig] = []

    @model_validator(mode="after")
    def _unique_names(self):
        names = [c.name for c in self.cameras]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate camera names: {sorted(dupes)}")
        return self


def load_settings(path: str | Path) -> Settings:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Settings.model_validate(raw)
