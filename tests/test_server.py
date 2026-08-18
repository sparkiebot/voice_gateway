import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from urllib import error, request

from voice_gateway.server import GatewayServer, _parse_multipart


def _multipart(boundary, fields, audio):
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.extend((f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="x.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
                  audio, f'\r\n--{boundary}--\r\n'.encode()))
    return b"".join(parts)


class MultipartTests(unittest.TestCase):
    content_type = "multipart/form-data; boundary=test"

    def test_requires_wav(self):
        body = (b"--test\r\nContent-Disposition: form-data; name=\"request_id\"\r\n\r\n"
                b"abc\r\n--test--\r\n")
        with self.assertRaisesRegex(ValueError, "audio"):
            _parse_multipart(self.content_type, body)

    def test_extracts_fields_and_audio(self):
        body = (b"--test\r\nContent-Disposition: form-data; name=\"request_id\"\r\n\r\n"
                b"abc\r\n--test\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"x.wav\"\r\n"
                b"Content-Type: audio/wav\r\n\r\nRIFFdata\r\n--test--\r\n")
        fields, audio = _parse_multipart(self.content_type, body)
        self.assertEqual(fields["request_id"], "abc")
        self.assertEqual(audio, b"RIFFdata")


class StubGateway:
    ready = True
    config = SimpleNamespace(shared_secret="secret", max_audio_bytes=1024 * 1024)
    def process(self, path: Path, request_id: str, robot_id: str, context, tools):
        self.received = (path.read_bytes(), request_id, robot_id, context, tools)
        return {"request_id": request_id, "type": "speech", "route": "mind_fallback",
                "reason": "conversational", "transcript": "ciao",
                "response_text": "Ciao!", "tool_calls": [], "executed": False,
                "timings_ms": {}}


class HttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.gateway = StubGateway()
        self.server = GatewayServer(("127.0.0.1", 0), self.gateway)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def test_health_requires_bearer_secret(self):
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(self.base + "/health")
        self.assertEqual(caught.exception.code, 401)
        req = request.Request(self.base + "/health",
                              headers={"Authorization": "Bearer secret"})
        with request.urlopen(req) as response:
            self.assertTrue(json.load(response)["ready"])

    def test_authenticated_voice_round_trip(self):
        fields = {"request_id": "request-9", "robot_id": "sparkie-01",
                  "language": "it", "timestamp": "now", "context": "{}",
                  "available_tools": "[]"}
        boundary = "integration-test"
        body = _multipart(boundary, fields, b"RIFF-original-wav")
        req = request.Request(self.base + "/v1/voice-requests", body,
            {"Authorization": "Bearer secret",
             "Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        with request.urlopen(req) as response:
            payload = json.load(response)
        self.assertEqual(payload["request_id"], "request-9")
        self.assertEqual(self.gateway.received[0], b"RIFF-original-wav")


if __name__ == "__main__": unittest.main()
