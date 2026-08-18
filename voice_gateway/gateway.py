"""Warm, single-inference orchestration with safe Mind fallback."""

from __future__ import annotations
import json
from pathlib import Path
import threading
import time
from typing import Any

from .audio import trim_silence, validate_wav
from .backends import NeedleRouter, NemoSpeechCuda
from .config import GatewayConfig
from .mind import MindClient, MindFailure
from .local_responses import LocalResponseCatalog
from .policy import evaluate
from .tool_catalog import ToolCatalog


class BusyError(RuntimeError):
    pass


class VoiceGateway:
    def __init__(self, config: GatewayConfig, asr: Any | None = None,
                 router: Any | None = None, mind: Any | None = None) -> None:
        self.config = config
        self.responses = LocalResponseCatalog(config.local_responses_path)
        self.catalog = ToolCatalog(config.tools_path, self.responses.intents)
        self.tools = self.catalog.all_tools
        self.local_tools = self.catalog.local_tools
        self.asr = asr or NemoSpeechCuda(str(config.nemo_model_path),
                                        str(config.nemo_library_path), config.language,
                                        config.nemo_gpu, config.nemo_warmup_seconds)
        self.router = router or NeedleRouter(self.catalog.needle_tools, config.needle_model_path)
        self.mind = mind or MindClient(config.mind_base_url, config.mind_timeout_seconds,
                                      config.robot_id, config.language,
                                      config.retry_count, config.retry_backoff_seconds)
        self._inference = threading.Lock()
        self._waiting = threading.BoundedSemaphore(config.queue_capacity) if config.queue_capacity else None
        self._ready = False
        self._cache_lock = threading.Lock()
        self._responses: dict[str, dict[str, Any]] = {}
        started = time.perf_counter()
        self.asr.initialize()
        self.initialization_ms = (time.perf_counter() - started) * 1000
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def process(self, wav_path: Path, request_id: str, robot_id: str,
                context: dict[str, Any], offered_tools: list[dict[str, Any]]) -> dict[str, Any]:
        if robot_id != self.config.robot_id:
            raise ValueError("robot_id_mismatch")
        if not request_id or len(request_id) > 128:
            raise ValueError("invalid_request_id")
        with self._cache_lock:
            cached = self._responses.get(request_id)
        if cached is not None:
            return cached
        admitted_waiter = False
        if not self._inference.acquire(blocking=False):
            if self._waiting is None or not self._waiting.acquire(blocking=False):
                raise BusyError("busy")
            admitted_waiter = True
            if not self._inference.acquire(timeout=self.config.request_timeout_seconds):
                self._waiting.release()
                raise BusyError("busy")
        if admitted_waiter:
            self._waiting.release()
        try:
            result = self._process_locked(wav_path, request_id, context, offered_tools)
            with self._cache_lock:
                if len(self._responses) >= 256:
                    self._responses.pop(next(iter(self._responses)))
                self._responses[request_id] = result
            return result
        finally:
            self._inference.release()

    def _process_locked(self, wav_path: Path, request_id: str, context: dict[str, Any],
                        offered_tools: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        robot_tools = self.catalog.offered_tools(offered_tools)
        route_tools = [*robot_tools, *self.local_tools]
        timings: dict[str, float] = {"initialization_ms": self.initialization_ms}
        transcript, proposal, reason = "", None, "local_error"
        try:
            phase = time.perf_counter()
            audio = validate_wav(wav_path, self.config.max_audio_seconds,
                                 self.config.max_audio_bytes)
            timings["audio_validation_ms"] = (time.perf_counter() - phase) * 1000
            asr_path = wav_path
            trimmed_path: Path | None = None
            phase = time.perf_counter()
            if self.config.trim_silence:
                trimmed = trim_silence(
                    wav_path,
                    self.config.trim_threshold_percent,
                    self.config.trim_leading_ms,
                    self.config.trim_trailing_ms,
                    self.config.trim_minimum_ms,
                )
                if trimmed is not None:
                    trimmed_path, trimmed_audio = trimmed
                    asr_path = trimmed_path
                    timings["asr_audio_duration_ms"] = (
                        trimmed_audio.duration_seconds * 1000
                    )
            timings["silence_trim_ms"] = (time.perf_counter() - phase) * 1000
            phase = time.perf_counter()
            try:
                asr_result = self.asr.transcribe(asr_path)
            finally:
                if trimmed_path is not None:
                    trimmed_path.unlink(missing_ok=True)
            asr_inference_ms = (time.perf_counter() - phase) * 1000
            timings["asr_inference_ms"] = asr_inference_ms
            transcript = str(asr_result.get("text", ""))
            if self.config.log_transcripts:
                _log_transcript(
                    request_id,
                    transcript,
                    asr_result.get("confidence"),
                    asr_inference_ms,
                )
            phase = time.perf_counter()
            needle_result = self.router.complete(transcript)
            timings["needle_inference_ms"] = (time.perf_counter() - phase) * 1000
            calls = needle_result.get("function_calls", needle_result.get("tool_calls", []))
            local_names = {tool.get("name") for tool in self.local_tools}
            proposed_name = (
                calls[0].get("name")
                if isinstance(calls, list) and len(calls) == 1
                and isinstance(calls[0], dict)
                else None
            )
            needle_threshold = (
                self.config.local_conversation_confidence_threshold
                if proposed_name in local_names
                else self.config.needle_confidence_threshold
            )
            route, reason, proposal = evaluate(
                asr_result, needle_result, route_tools, self.config.asr_confidence_threshold,
                needle_threshold, self.config.sensitive_tools,
                self.config.allow_unscored_needle)
            if route == "local_proposal":
                local_spec = next(
                    (tool for tool in self.local_tools
                     if proposal and tool.get("name") == proposal.get("name")),
                    None,
                )
                if local_spec is not None:
                    intent_argument = local_spec.get("response_intent_argument")
                    intent = str(
                        proposal["arguments"].get(intent_argument, "")
                        if isinstance(intent_argument, str) and proposal else
                        local_spec.get("local_intent", "")
                    )
                    if (
                        intent in self.responses.intents
                        and self.responses.matches(transcript, intent)
                    ):
                        timings["request_total_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                        result = _response(
                            request_id, "local_response",
                            "validated_simple_conversation", transcript,
                            self.responses.choose(intent), None, timings,
                            executed=False,
                        )
                        _log_result(result)
                        return result
                    route, reason, proposal = (
                        "mind_fallback", "local_intent_phrase_mismatch", None
                    )
                else:
                    timings["request_total_ms"] = (
                        time.perf_counter() - started
                    ) * 1000
                    result = _response(
                        request_id, route, reason, transcript, "", proposal,
                        timings, executed=False,
                    )
                    _log_result(result)
                    return result
        except Exception as exc:  # CUDA/router/audio boundary always fails toward Mind.
            reason = f"local_{type(exc).__name__}"
            proposal = None

        try:
            mind_result, mind_timings = self.mind.submit(
                transcript, request_id, context, robot_tools)
            timings.update(mind_timings)
            timings["request_total_ms"] = (time.perf_counter() - started) * 1000
            result = _response(request_id, "mind_fallback", reason, transcript,
                               mind_result["response_text"], None, timings,
                               executed=False, tool_proposals=[])
            _log_result(result)
            return result
        except MindFailure as exc:
            timings["request_total_ms"] = (time.perf_counter() - started) * 1000
            result = _response(request_id, "error", f"{reason}:{exc.reason}", transcript,
                               "", None, timings, executed=False, error=exc.reason)
            _log_result(result)
            return result

    def close(self) -> None:
        self._ready = False
        close = getattr(self.asr, "close", None)
        if close:
            close()


def _response(request_id: str, route: str, reason: str, transcript: str,
              response_text: str, proposal: dict[str, Any] | None,
              timings: dict[str, float], executed: bool,
              tool_proposals: list[dict[str, Any]] | None = None,
              error: str | None = None) -> dict[str, Any]:
    proposals = tool_proposals if tool_proposals is not None else ([proposal] if proposal else [])
    return {"request_id": request_id, "type": "tool_call" if proposals else "speech",
            "route": route, "reason": reason, "transcript": transcript,
            "response_text": response_text, "tool_calls": proposals,
            "executed": executed, "timings_ms": {key: round(value, 1) for key, value in timings.items()},
            **({"error": error} if error else {})}


def _log_transcript(
    request_id: str,
    transcript: str,
    confidence: Any,
    asr_inference_ms: float,
) -> None:
    safe_transcript = "".join(
        character if character.isprintable() else " " for character in transcript
    )[:1000]
    print(json.dumps({
        "event": "voice_transcript",
        "request_id": request_id,
        "transcript": safe_transcript,
        "confidence": confidence if isinstance(confidence, (int, float)) else None,
        "asr_inference_ms": round(asr_inference_ms, 1),
    }, ensure_ascii=False, separators=(",", ":")), flush=True)


def _log_result(result: dict[str, Any]) -> None:
    print(json.dumps({
        "event": "voice_request_complete",
        "request_id": result["request_id"],
        "route": result["route"],
        "reason": result["reason"],
        "timings_ms": result["timings_ms"],
    }, ensure_ascii=False, separators=(",", ":")), flush=True)
