"""Validate model output as a proposal; this module never executes tools."""

from __future__ import annotations
from typing import Any


def _value_errors(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    kind = schema.get("type")
    expected = {"string": str, "object": dict, "integer": int,
                "number": (int, float), "boolean": bool}.get(kind)
    if expected and (not isinstance(value, expected) or
                     (kind in {"integer", "number"} and isinstance(value, bool))):
        return [f"{path} must be {kind}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} is not allowed")
    if kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{key} is not allowed" for key in value if key not in properties)
        for key, child in value.items():
            if key in properties:
                errors.extend(_value_errors(child, properties[key], f"{path}.{key}"))
    return errors


def evaluate(asr: dict[str, Any], needle: dict[str, Any], tools: list[dict[str, Any]],
             asr_threshold: float, needle_threshold: float,
             sensitive: frozenset[str], allow_unscored_needle: bool = False
             ) -> tuple[str, str, dict[str, Any] | None]:
    text = str(asr.get("text", "")).strip()
    if not text:
        return "mind_fallback", "empty_transcript", None
    confidence = asr.get("confidence")
    if confidence is not None and float(confidence) < asr_threshold:
        return "mind_fallback", "low_asr_confidence", None
    if not needle.get("success"):
        return "mind_fallback", "needle_failed", None
    confidence = needle.get("confidence")
    if confidence is None and not allow_unscored_needle:
        return "mind_fallback", "missing_needle_confidence", None
    if confidence is not None and float(confidence) < needle_threshold:
        return "mind_fallback", "low_needle_confidence", None
    calls = needle.get("function_calls", needle.get("tool_calls", []))
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return "mind_fallback", "conversational_or_malformed", None
    call = calls[0]
    name, arguments = call.get("name"), call.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return "mind_fallback", "malformed_tool_call", None
    spec = next((item for item in tools if item.get("name") == name), None)
    if spec is None:
        return "mind_fallback", "tool_not_permitted", None
    if name in sensitive:
        return "mind_fallback", "sensitive_tool", None
    if _value_errors(arguments, spec.get("parameters", {}), "arguments"):
        return "mind_fallback", "invalid_tool_arguments", None
    return "local_proposal", "validated_non_sensitive_proposal", {
        "name": name, "arguments": arguments, "status": "proposed"
    }
