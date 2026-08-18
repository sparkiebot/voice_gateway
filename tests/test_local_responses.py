import json
from pathlib import Path
import tempfile
import unittest

from voice_gateway.local_responses import (
    LocalResponseCatalog,
    matches_local_intent,
)


class FirstChoice:
    def choice(self, values): return values[0]


class LocalResponseCatalogTests(unittest.TestCase):
    def test_phrase_validation_is_intent_specific(self):
        self.assertTrue(matches_local_intent("Come stai?", "wellbeing"))
        self.assertFalse(matches_local_intent("Come stai, Sparkie?", "wellbeing"))
        self.assertTrue(matches_local_intent("Grazie mille!", "thanks"))
        self.assertFalse(matches_local_intent("Chi sei?", "thanks"))
        self.assertFalse(matches_local_intent("Accendi la luce", "greeting"))

    def test_consecutive_response_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.json"
            path.write_text(json.dumps({"greeting": ["one", "two", "three"]}))
            catalog = LocalResponseCatalog(path, FirstChoice())
            self.assertEqual(catalog.choose("greeting"), "one")
            self.assertEqual(catalog.choose("greeting"), "two")
            self.assertEqual(catalog.choose("greeting"), "one")

    def test_unknown_intent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.json"
            path.write_text(json.dumps({"greeting": ["one", "two"]}))
            catalog = LocalResponseCatalog(path)
            with self.assertRaisesRegex(ValueError, "unknown"):
                catalog.choose("unsafe")

    def test_object_catalogue_owns_phrase_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.json"
            path.write_text(json.dumps({"wake": {
                "phrases": ["Sveglia!"],
                "responses": ["Eccomi", "Sono sveglio"],
            }}))
            catalog = LocalResponseCatalog(path)
            self.assertTrue(catalog.matches("sveglia", "wake"))
            self.assertFalse(catalog.matches("sveglia sparkie", "wake"))


if __name__ == "__main__": unittest.main()
