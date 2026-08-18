"""Private-LAN Mind text fallback preserving the gateway request identity."""

from __future__ import annotations
from datetime import datetime, timezone
import json
import socket
import time
from typing import Any
from urllib import error, request


class MindFailure(RuntimeError):
    def __init__(self, reason: str, retryable: bool = False) -> None:
        super().__init__(reason)
        self.reason, self.retryable = reason, retryable


class MindClient:
    def __init__(self, base_url: str, timeout: float, robot_id: str, language: str,
                 retry_count: int, retry_backoff: float) -> None:
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.robot_id, self.language = robot_id, language
        self.retry_count, self.retry_backoff = retry_count, retry_backoff

    def submit(self, text: str, request_id: str, context: dict[str, Any],
               tools: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
        total_started = time.perf_counter()
        total_health = total_http = total_backoff = 0.0
        for attempt in range(self.retry_count + 1):
            started = time.perf_counter()
            try:
                result, health_ms, http_ms = self._submit_once(
                    text, request_id, context, tools
                )
                total_health += health_ms
                total_http += http_ms
                return result, {
                    "mind_health_ms": total_health,
                    "mind_http_ms": total_http,
                    "mind_retry_backoff_ms": total_backoff,
                    "mind_total_ms": (time.perf_counter() - total_started) * 1000,
                }
            except MindFailure as exc:
                total_http += (time.perf_counter() - started) * 1000
                if not exc.retryable or attempt == self.retry_count:
                    raise
                delay = self.retry_backoff * (2 ** attempt)
                time.sleep(delay)
                total_backoff += delay * 1000
        raise AssertionError("unreachable")

    def _submit_once(self, text: str, request_id: str, context: dict[str, Any],
                     tools: list[dict[str, Any]]) -> tuple[dict[str, Any], float, float]:
        try:
            health_started = time.perf_counter()
            with request.urlopen(self.base_url + "/health", timeout=self.timeout) as response:
                health = json.load(response)
            health_ms = (time.perf_counter() - health_started) * 1000
            if not isinstance(health, dict) or health.get("ready") is not True:
                raise MindFailure("mind_not_ready")
            body = json.dumps({
                "request_id": request_id,
                "robot_id": self.robot_id,
                "language": self.language,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "text": text,
                "context": context,
                "available_tools": tools,
            }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            req = request.Request(self.base_url + "/v1/text-requests", body, {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }, method="POST")
            http_started = time.perf_counter()
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = json.load(response)
            http_ms = (time.perf_counter() - http_started) * 1000
        except error.HTTPError as exc:
            raise MindFailure("mind_busy" if exc.code == 429 else f"mind_http_{exc.code}",
                              retryable=exc.code == 429) from exc
        except (error.URLError, TimeoutError, socket.timeout, OSError, ValueError,
                json.JSONDecodeError) as exc:
            raise MindFailure("mind_unavailable") from exc
        if not isinstance(payload, dict) or payload.get("request_id") != request_id:
            raise MindFailure("mind_invalid_response")
        if not isinstance(payload.get("response_text"), str):
            raise MindFailure("mind_invalid_response")
        calls = payload.get("tool_calls", [])
        if not isinstance(calls, list):
            raise MindFailure("mind_invalid_response")
        permitted = {tool.get("name") for tool in tools}
        for call in calls:
            if (not isinstance(call, dict) or call.get("name") not in permitted
                    or not isinstance(call.get("arguments"), dict)):
                raise MindFailure("mind_invalid_tool_proposal")
        return payload, health_ms, http_ms
