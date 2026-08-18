"""Canonical, validated tool catalogue used at every gateway boundary."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


_GATEWAY_FIELDS = frozenset({
    "gateway_local", "local_intent", "response_intent_argument",
})


class ToolCatalog:
    """Loads tool contracts once; callers may only select their availability."""

    def __init__(self, path: Path, local_intents: frozenset[str]) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("tools_path must contain a non-empty JSON list")
        self._by_name: dict[str, dict[str, Any]] = {}
        for tool in payload:
            self._validate(tool, local_intents)
            name = tool["name"]
            if name in self._by_name:
                raise ValueError(f"duplicate tool name: {name}")
            self._by_name[name] = tool

    @staticmethod
    def _validate(tool: Any, local_intents: frozenset[str]) -> None:
        if not isinstance(tool, dict):
            raise ValueError("every tool must be a JSON object")
        name, description, parameters = (
            tool.get("name"), tool.get("description"), tool.get("parameters")
        )
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("tool name must be a non-empty snake_case identifier")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"tool {name!r} needs a description")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError(f"tool {name!r} needs an object parameters schema")
        if not isinstance(parameters.get("properties", {}), dict):
            raise ValueError(f"tool {name!r} has invalid properties")
        if not isinstance(parameters.get("required", []), list):
            raise ValueError(f"tool {name!r} has invalid required fields")
        if tool.get("gateway_local") is True:
            intent = tool.get("local_intent")
            intent_argument = tool.get("response_intent_argument")
            if isinstance(intent, str) and intent in local_intents and intent_argument is None:
                return
            if not isinstance(intent_argument, str) or intent is not None:
                raise ValueError(
                    f"local tool {name!r} needs local_intent or response_intent_argument"
                )
            argument_schema = parameters.get("properties", {}).get(intent_argument)
            choices = argument_schema.get("enum") if isinstance(argument_schema, dict) else None
            if (intent_argument not in parameters.get("required", [])
                    or not isinstance(choices, list) or not choices
                    or any(choice not in local_intents for choice in choices)):
                raise ValueError(f"local tool {name!r} has invalid response intent argument")
        elif "local_intent" in tool or "response_intent_argument" in tool:
            raise ValueError(f"non-local tool {name!r} cannot define local response metadata")

    @property
    def all_tools(self) -> list[dict[str, Any]]:
        return list(self._by_name.values())

    @property
    def local_tools(self) -> list[dict[str, Any]]:
        return [tool for tool in self._by_name.values() if tool.get("gateway_local") is True]

    @property
    def needle_tools(self) -> list[dict[str, Any]]:
        return [{key: value for key, value in tool.items() if key not in _GATEWAY_FIELDS}
                for tool in self._by_name.values()]

    def offered_tools(self, offered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return canonical remote tools named by the robot; ignore caller schemas."""
        names = {item.get("name") for item in offered if isinstance(item, dict)}
        return [tool for name, tool in self._by_name.items()
                if name in names and tool.get("gateway_local") is not True]
