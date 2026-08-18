"""Configuration loading and validation for the host gateway."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class GatewayConfig:
    bind_address: str
    port: int
    shared_secret: str
    robot_id: str
    language: str
    mind_base_url: str
    request_timeout_seconds: float
    mind_timeout_seconds: float
    retry_count: int
    retry_backoff_seconds: float
    queue_capacity: int
    max_audio_seconds: float
    max_audio_bytes: int
    nemo_model_path: Path
    nemo_library_path: Path
    nemo_gpu: int
    nemo_warmup_seconds: float
    needle_model_path: str
    tools_path: Path
    local_responses_path: Path
    asr_confidence_threshold: float
    needle_confidence_threshold: float
    local_conversation_confidence_threshold: float
    sensitive_tools: frozenset[str]
    log_transcripts: bool
    trim_silence: bool
    trim_threshold_percent: float
    trim_leading_ms: int
    trim_trailing_ms: int
    trim_minimum_ms: int
    allow_unscored_needle: bool = False
    needle_system: str = "locale: it-IT; assistant: Sparkie"

    @classmethod
    def load(cls, path: Path) -> "GatewayConfig":
        source = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError("gateway configuration must be a JSON object")
        root = path.resolve().parent

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

        secret = os.environ.get("SPARKIE_VOICE_GATEWAY_SECRET", source.get("shared_secret", ""))
        config = cls(
            bind_address=str(source.get("bind_address", "127.0.0.1")),
            port=int(source.get("port", 8090)),
            shared_secret=str(secret),
            robot_id=str(source.get("robot_id", "sparkie-01")),
            language=str(source.get("language", "it")),
            mind_base_url=str(source.get("mind_base_url", "")),
            request_timeout_seconds=float(source.get("request_timeout_seconds", 25.0)),
            mind_timeout_seconds=float(source.get("mind_timeout_seconds", 20.0)),
            retry_count=int(source.get("retry_count", 1)),
            retry_backoff_seconds=float(source.get("retry_backoff_seconds", 0.5)),
            queue_capacity=int(source.get("queue_capacity", 0)),
            max_audio_seconds=float(source.get("max_audio_seconds", 20.0)),
            max_audio_bytes=int(source.get("max_audio_bytes", 8 * 1024 * 1024)),
            nemo_model_path=resolve(str(source["nemo_model_path"])),
            nemo_library_path=resolve(str(source["nemo_library_path"])),
            nemo_gpu=int(source.get("nemo_gpu", 0)),
            nemo_warmup_seconds=float(source.get("nemo_warmup_seconds", 2.0)),
            needle_model_path=str(resolve(str(source["needle_model_path"]))) if source.get("needle_model_path") else "",
            tools_path=resolve(str(source.get("tools_path", "tools.json"))),
            local_responses_path=resolve(str(source.get(
                "local_responses_path", "responses.it.json"
            ))),
            asr_confidence_threshold=float(source.get("asr_confidence_threshold", 0.70)),
            needle_confidence_threshold=float(source.get("needle_confidence_threshold", 0.85)),
            local_conversation_confidence_threshold=float(source.get(
                "local_conversation_confidence_threshold", 0.75
            )),
            sensitive_tools=frozenset(map(str, source.get("sensitive_tools", []))),
            log_transcripts=bool(source.get("log_transcripts", False)),
            trim_silence=bool(source.get("trim_silence", True)),
            trim_threshold_percent=float(source.get("trim_threshold_percent", 0.5)),
            trim_leading_ms=int(source.get("trim_leading_ms", 100)),
            trim_trailing_ms=int(source.get("trim_trailing_ms", 150)),
            trim_minimum_ms=int(source.get("trim_minimum_ms", 300)),
            allow_unscored_needle=bool(source.get("allow_unscored_needle", False)),
            needle_system=str(source.get("needle_system", "locale: it-IT; assistant: Sparkie")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.shared_secret:
            raise ValueError("shared secret is required (prefer SPARKIE_VOICE_GATEWAY_SECRET)")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.queue_capacity not in (0, 1):
            raise ValueError("queue_capacity must be 0 or 1")
        if self.retry_count < 0 or self.retry_backoff_seconds < 0:
            raise ValueError("retry settings cannot be negative")
        if min(self.request_timeout_seconds, self.mind_timeout_seconds,
               self.max_audio_seconds) <= 0 or self.max_audio_bytes <= 44:
            raise ValueError("timeouts and audio limits must be positive")
        if not 0 < self.trim_threshold_percent <= 100:
            raise ValueError("trim_threshold_percent must be between 0 and 100")
        if min(self.trim_leading_ms, self.trim_trailing_ms,
               self.trim_minimum_ms) < 0:
            raise ValueError("trim durations cannot be negative")
        if not 0 <= self.local_conversation_confidence_threshold <= 1:
            raise ValueError(
                "local_conversation_confidence_threshold must be between 0 and 1"
            )
        parsed = urlsplit(self.mind_base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("mind_base_url must be an HTTP(S) URL")
