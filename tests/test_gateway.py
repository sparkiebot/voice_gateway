import json
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
import wave

from voice_gateway.config import GatewayConfig
from voice_gateway.gateway import BusyError, VoiceGateway


class StubASR:
    def __init__(self, text="quanta batteria rimane", confidence=0.99, delay=0):
        self.text, self.confidence, self.delay = text, confidence, delay
        self.received_frames = None
    def initialize(self): pass
    def transcribe(self, _path):
        time.sleep(self.delay)
        with wave.open(str(_path), "rb") as source:
            self.received_frames = source.getnframes()
        return {"text": self.text, "confidence": self.confidence}


class StubNeedle:
    def __init__(self, result): self.result = result
    def complete(self, _text): return self.result


class StubMind:
    def __init__(self): self.calls = []
    def submit(self, text, request_id, context, tools):
        self.calls.append((text, request_id, context, tools))
        return {"request_id": request_id, "response_text": "Ciao",
                "tool_calls": [{"name": "get_battery_level", "arguments": {}}]}, \
               {"mind_http_ms": 3.0, "retry_backoff_ms": 0.0}


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.local_tool = {
            "name": "smalltalk_wellbeing", "description": "Answer a harmless wellbeing question.", "gateway_local": True,
            "local_intent": "wellbeing",
            "parameters": {"type": "object", "properties": {},
                           "required": [], "additionalProperties": False}}
        self.tools = [self.local_tool, {"name": "get_battery_level", "description": "Read the battery level.", "parameters": {
            "type": "object", "properties": {}, "additionalProperties": False}}]
        tools_path = root / "tools.json"
        tools_path.write_text(json.dumps(self.tools))
        responses_path = root / "responses.json"
        responses_path.write_text(json.dumps({
            "greeting": {"phrases": ["ciao"], "responses": ["Ciao uno", "Ciao due"]},
            "wellbeing": {"phrases": ["come va"], "responses": ["Bene uno", "Bene due"]},
        }))
        self.wav = root / "request.wav"
        with wave.open(str(self.wav), "wb") as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
            output.writeframes(b"\x01\x00" * 1600)
        self.config = GatewayConfig(
            "127.0.0.1", 8090, "secret", "sparkie-01", "it",
            "http://127.0.0.1:8088", 1, 1, 0, 0, 0, 20, 999999,
            root / "model", root / "library", 0, 0, "", tools_path,
            responses_path,
            .7, .85, .75, frozenset({"navigate_to"}), True,
            True, .5, 100, 150, 300)

    def tearDown(self): self.temporary.cleanup()

    def test_local_proposal_is_never_executed(self):
        mind = StubMind()
        gateway = VoiceGateway(self.config, StubASR(), StubNeedle({
            "success": True, "confidence": .95,
            "function_calls": [{"name": "get_battery_level", "arguments": {}}]}), mind)
        result = gateway.process(self.wav, "request-1", "sparkie-01", {}, self.tools)
        self.assertEqual(result["route"], "local_proposal")
        self.assertFalse(result["executed"])
        self.assertEqual(mind.calls, [])

    def test_simple_conversation_returns_local_speech_without_tool(self):
        mind = StubMind()
        gateway = VoiceGateway(self.config, StubASR(text="come va"), StubNeedle({
            "success": True, "confidence": .98,
            "function_calls": [{"name": "smalltalk_wellbeing",
                                "arguments": {}}]}), mind)
        result = gateway.process(self.wav, "small-talk", "sparkie-01", {}, [])
        self.assertEqual(result["route"], "local_response")
        self.assertEqual(result["reason"], "validated_simple_conversation")
        self.assertIn(result["response_text"], {"Bene uno", "Bene due"})
        self.assertEqual(result["tool_calls"], [])
        self.assertFalse(result["executed"])
        self.assertEqual(mind.calls, [])

    def test_dynamic_smalltalk_intent_is_validated_against_the_phrase(self):
        self.tools[0] = {
            "name": "local_smalltalk", "description": "Classify harmless small talk.",
            "gateway_local": True, "response_intent_argument": "intent",
            "parameters": {"type": "object", "properties": {
                "intent": {"type": "string", "enum": ["greeting", "wellbeing"]},
            }, "required": ["intent"], "additionalProperties": False},
        }
        (Path(self.temporary.name) / "tools.json").write_text(json.dumps(self.tools))
        mind = StubMind()
        gateway = VoiceGateway(self.config, StubASR(text="come va"), StubNeedle({
            "success": True, "confidence": .98,
            "function_calls": [{"name": "local_smalltalk", "arguments": {
                "intent": "wellbeing",
            }}],
        }), mind)
        result = gateway.process(self.wav, "small-talk-dynamic", "sparkie-01", {}, [])
        self.assertEqual(result["route"], "local_response")
        self.assertEqual(mind.calls, [])

    def test_transcript_log_is_structured_and_sanitized(self):
        gateway = VoiceGateway(self.config, StubASR(text="ciao\nrobot"),
            StubNeedle({"success": False}), StubMind())
        captured = io.StringIO()
        with redirect_stdout(captured):
            gateway.process(self.wav, "logged-id", "sparkie-01", {}, [])
        events = [json.loads(line) for line in captured.getvalue().splitlines()]
        event = next(item for item in events if item["event"] == "voice_transcript")
        self.assertEqual(event["event"], "voice_transcript")
        self.assertEqual(event["request_id"], "logged-id")
        self.assertEqual(event["transcript"], "ciao robot")
        self.assertIsInstance(event["asr_inference_ms"], float)
        self.assertGreaterEqual(event["asr_inference_ms"], 0.0)
        complete = next(
            item for item in events if item["event"] == "voice_request_complete"
        )
        self.assertEqual(complete["request_id"], "logged-id")
        self.assertEqual(complete["route"], "mind_fallback")
        self.assertIn("mind_http_ms", complete["timings_ms"])

    def test_conversation_falls_back_with_transcript_and_id(self):
        mind = StubMind()
        gateway = VoiceGateway(self.config, StubASR(), StubNeedle({
            "success": True, "confidence": .99, "function_calls": []}), mind)
        result = gateway.process(self.wav, "request-2", "sparkie-01", {}, self.tools)
        self.assertEqual(result["route"], "mind_fallback")
        self.assertEqual(result["tool_calls"], [])  # fallback proposals cannot execute
        self.assertEqual(mind.calls[0][0:2], ("quanta batteria rimane", "request-2"))

    def test_offered_tool_schema_cannot_override_canonical_contract(self):
        mind = StubMind()
        gateway = VoiceGateway(self.config, StubASR(), StubNeedle({"success": False}), mind)
        gateway.process(self.wav, "canonical-tools", "sparkie-01", {}, [{
            "name": "get_battery_level", "description": "untrusted", "parameters": {"type": "string"}
        }])
        self.assertEqual(mind.calls[0][3], [self.tools[1]])

    def test_unscored_needle_fails_closed_by_default(self):
        mind = StubMind()
        gateway = VoiceGateway(self.config, StubASR(), StubNeedle({
            "success": True, "confidence": None,
            "function_calls": [{"name": "get_battery_level", "arguments": {}}],
        }), mind)
        result = gateway.process(self.wav, "no-score", "sparkie-01", {}, self.tools)
        self.assertEqual(result["route"], "mind_fallback")
        self.assertEqual(result["reason"], "missing_needle_confidence")

    def test_asr_copy_is_trimmed_and_mind_receives_transcript(self):
        with wave.open(str(self.wav), "wb") as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
            output.writeframes(b"\x00\x00" * 3200)
            output.writeframes(b"\x00\x20" * 6400)
            output.writeframes(b"\x00\x00" * 4800)
        asr, mind = StubASR(), StubMind()
        gateway = VoiceGateway(self.config, asr, StubNeedle({"success": False}), mind)
        result = gateway.process(self.wav, "trimmed", "sparkie-01", {}, [])
        self.assertLess(asr.received_frames, 14400)
        self.assertEqual(asr.received_frames, 10400)
        self.assertEqual(mind.calls[0][0], "quanta batteria rimane")
        self.assertIn("silence_trim_ms", result["timings_ms"])
        self.assertEqual(result["timings_ms"]["asr_audio_duration_ms"], 650.0)
        self.assertIn("request_total_ms", result["timings_ms"])
        self.assertNotIn("total_ms", result["timings_ms"])

    def test_completed_request_is_deduplicated(self):
        mind = StubMind()
        gateway = VoiceGateway(self.config, StubASR(), StubNeedle({"success": False}), mind)
        first = gateway.process(self.wav, "same", "sparkie-01", {}, [])
        second = gateway.process(self.wav, "same", "sparkie-01", {}, [])
        self.assertEqual(first, second)
        self.assertEqual(len(mind.calls), 1)

    def test_zero_queue_rejects_concurrent_request(self):
        gateway = VoiceGateway(self.config, StubASR(delay=.15),
                               StubNeedle({"success": False}), StubMind())
        worker = threading.Thread(target=gateway.process,
            args=(self.wav, "first", "sparkie-01", {}, []))
        worker.start(); time.sleep(.03)
        with self.assertRaises(BusyError):
            gateway.process(self.wav, "second", "sparkie-01", {}, [])
        worker.join()

    def test_queue_capacity_over_one_is_rejected(self):
        values = dict(self.config.__dict__); values["queue_capacity"] = 2
        with self.assertRaisesRegex(ValueError, "queue_capacity"):
            GatewayConfig(**values).validate()


if __name__ == "__main__": unittest.main()
