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
        for intent, responses in payload.items():
            if (
                not isinstance(intent, str)
                or not isinstance(responses, list)
                or len(responses) < 2
                or any(not isinstance(response, str) or not response.strip()
                       for response in responses)
            ):
                raise ValueError(f"invalid local response catalog entry: {intent!r}")
            self._responses[intent] = tuple(response.strip() for response in responses)
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


_INTENT_PATTERNS = {
    "greeting": (r"^(ciao|salve|ehi|hey)( sparkie)?$",),
    "wellbeing": (r"^come (va|stai)( sparkie)?$", r"^tutto bene( sparkie)?$"),
    "thanks": (r"^(grazie|grazie mille|ti ringrazio)( sparkie)?$",),
    "goodbye": (r"^(arrivederci|a presto|a dopo|ci vediamo)( sparkie)?$",),
    "identity": (r"^(chi sei|come ti chiami)( sparkie)?$",),
    "capabilities": (r"^(cosa|che cosa) (sai|puoi) fare( sparkie)?$",),
    "compliment_response": (
        r"^(sei|quanto sei) (bravo|bravissimo|forte|simpatico)( sparkie)?$",
    ),
    "encouragement": (
        r"^(incoraggiami|motivami|dammi coraggio)( sparkie)?$",
    ),
    "good_morning": (r"^buongiorno( sparkie)?$",),
    "good_evening": (r"^buonasera( sparkie)?$",),
    "good_night": (r"^(buonanotte|buon riposo|notte)( sparkie)?$",),
}


def matches_local_intent(text: str, intent: str) -> bool:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return any(re.fullmatch(pattern, normalized) for pattern in _INTENT_PATTERNS.get(intent, ()))
