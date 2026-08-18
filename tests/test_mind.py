from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from voice_gateway.mind import MindClient


class MindStubHandler(BaseHTTPRequestHandler):
    request_path = ""
    request_content_type = ""
    request_payload = None

    def do_GET(self):
        self._send({"ready": True})

    def do_POST(self):
        type(self).request_path = self.path
        type(self).request_content_type = self.headers.get("Content-Type", "")
        length = int(self.headers["Content-Length"])
        type(self).request_payload = json.loads(self.rfile.read(length))
        self._send({"request_id": self.request_payload["request_id"],
                    "response_text": "Risposta Mind", "tool_calls": []})

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args): pass


class MindTextContractTests(unittest.TestCase):
    def setUp(self):
        MindStubHandler.request_payload = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), MindStubHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def test_fallback_posts_json_transcript_without_audio(self):
        base_url = f"http://127.0.0.1:{self.server.server_port}"
        client = MindClient(base_url, 1, "sparkie-01", "it", 0, 0)
        result, timings = client.submit(
            "accendi la luce", "request-text-1", {"battery": 80}, [])
        self.assertEqual(result["response_text"], "Risposta Mind")
        self.assertEqual(MindStubHandler.request_path, "/v1/text-requests")
        self.assertEqual(MindStubHandler.request_content_type, "application/json")
        self.assertEqual(MindStubHandler.request_payload["text"], "accendi la luce")
        self.assertEqual(MindStubHandler.request_payload["context"], {"battery": 80})
        self.assertNotIn("audio", MindStubHandler.request_payload)
        self.assertIn("mind_health_ms", timings)
        self.assertIn("mind_http_ms", timings)
        self.assertIn("mind_total_ms", timings)


if __name__ == "__main__": unittest.main()
