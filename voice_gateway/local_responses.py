"""Deterministic-boundary random responses for safe local conversation."""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
import unicodedata
from typing import Any


class LocalResponseCatalog:
    def __init__(self, path: Path, rng: Any | None = None) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError("local response catalog must be a non-empty object")
        self._responses: dict[str, tuple[str, ...]] = {}
        self._phrases: dict[str, frozenset[str]] = {}
        for intent, item in payload.items():
            # The object form keeps all routing phrases next to their answers.
            # A bare response list remains accepted for backward compatibility.
            responses = item.get("responses") if isinstance(item, dict) else item
            phrases = item.get("phrases", []) if isinstance(item, dict) else []
            if (
                not isinstance(intent, str)
                or not isinstance(responses, list)
                or len(responses) < 2
                or any(not isinstance(response, str) or not response.strip()
                       for response in responses)
            ):
                raise ValueError(f"invalid local response catalog entry: {intent!r}")
            self._responses[intent] = tuple(response.strip() for response in responses)
            if not isinstance(phrases, list) or any(
                not isinstance(phrase, str) or not phrase.strip() for phrase in phrases
            ):
                raise ValueError(f"invalid local response phrases: {intent!r}")
            self._phrases[intent] = frozenset(_normalize(phrase) for phrase in phrases)
        self._rng = rng or random.SystemRandom()
        self._last: dict[str, str] = {}

    @property
    def intents(self) -> frozenset[str]:
        return frozenset(self._responses)

    def choose(self, intent: str) -> str:
        responses = self._responses.get(intent)
        if responses is None:
            raise ValueError("unknown local conversation intent")
        previous = self._last.get(intent)
        candidates = [response for response in responses if response != previous]
        selected = self._rng.choice(candidates)
        self._last[intent] = selected
        return selected

    def matches(self, text: str, intent: str) -> bool:
        if intent not in self._responses:
            return False
        phrases = self._phrases[intent]
        if phrases:
            return _normalize(text) in phrases
        # Legacy catalogues did not carry phrases. Keep them working while new
        # catalogues use the data-driven form above.
        return matches_local_intent(text, intent)


_LEGACY_INTENT_PATTERNS = {
    "greeting": (r"^(ciao|salve|ehi|hey)$",),
    "wellbeing": (r"^come (va|stai)$", r"^tutto bene$"),
    "thanks": (r"^(grazie|grazie mille|ti ringrazio)$",),
    "goodbye": (r"^(arrivederci|a presto|a dopo|ci vediamo)$",),
    "identity": (r"^(chi sei|come ti chiami)$",),
    "capabilities": (r"^(cosa|che cosa) (sai|puoi) fare$",),
    "compliment_response": (
        r"^(sei|quanto sei) (bravo|bravissimo|forte|simpatico)$",
    ),
    "encouragement": (
        r"^(incoraggiami|motivami|dammi coraggio)$",
    ),
    "good_morning": (r"^buongiorno$",),
    "good_evening": (r"^buonasera$",),
    "good_night": (r"^(buonanotte|buon riposo|notte)$",),
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return normalized


def matches_local_intent(text: str, intent: str) -> bool:
    """Compatibility helper for users of pre-object response catalogues."""
    return any(re.fullmatch(pattern, _normalize(text))
               for pattern in _LEGACY_INTENT_PATTERNS.get(intent, ()))
