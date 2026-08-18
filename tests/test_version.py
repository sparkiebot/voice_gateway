import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from voice_gateway.version import source_version


class SourceVersionTests(unittest.TestCase):
    def test_environment_override_wins(self) -> None:
        with patch.dict(os.environ, {"VOICE_GATEWAY_SOURCE_VERSION": "release-7"}):
            self.assertEqual(source_version(Path("/missing")), "release-7")

    def test_non_repository_is_unknown(self) -> None:
        with patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as path:
            self.assertEqual(source_version(Path(path)), "unknown")


if __name__ == "__main__":
    unittest.main()
