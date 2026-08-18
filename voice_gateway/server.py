"""Authenticated HTTP server for the container-to-host voice boundary."""

from __future__ import annotations
import argparse
from email.parser import BytesParser
from email.policy import default
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any

from .config import GatewayConfig
from .gateway import BusyError, VoiceGateway
from .version import source_version


SOURCE_VERSION = source_version()


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], gateway: VoiceGateway) -> None:
        self.gateway = gateway
        super().__init__(address, RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server: GatewayServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(404, {"error": "not_found"})
        elif not self._authorized():
            self._json(401, {"error": "unauthorized"})
        else:
            self._json(200 if self.server.gateway.ready else 503,
                       {"ready": self.server.gateway.ready,
                        "service": "voice_gateway",
                        "version": SOURCE_VERSION})

    def do_POST(self) -> None:
        if self.path != "/v1/voice-requests":
            self._json(404, {"error": "not_found"})
            return
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        maximum = self.server.gateway.config.max_audio_bytes + 128 * 1024
        if length <= 0 or length > maximum:
            self._json(413, {"error": "invalid_request"})
            return
        temporary: Path | None = None
        try:
            fields, audio = _parse_multipart(self.headers.get("Content-Type", ""),
                                             self.rfile.read(length))
            descriptor, name = tempfile.mkstemp(prefix="sparkie-voice-", suffix=".wav")
            with os.fdopen(descriptor, "wb") as output:
                output.write(audio)
            temporary = Path(name)
            os.chmod(temporary, 0o600)
            context = _json_field(fields, "context", dict)
            tools = _json_field(fields, "available_tools", list)
            result = self.server.gateway.process(
                temporary, fields["request_id"], fields["robot_id"], context, tools)
            self._json(502 if result.get("route") == "error" else 200, result)
        except BusyError:
            self._json(429, {"error": "busy"}, retry_after="1")
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            self._json(400, {"error": "invalid_request"})
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _authorized(self) -> bool:
        expected = "Bearer " + self.server.gateway.config.shared_secret
        return hmac.compare_digest(self.headers.get("Authorization", ""), expected)

    def _json(self, status: int, payload: dict[str, Any], retry_after: str | None = None) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if retry_after:
            self.send_header("Retry-After", retry_after)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: object) -> None:
        # Never log request bodies, audio, context, or credentials.
        print(f"voice_gateway client={self.client_address[0]} {format_string % args}")


def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], bytes]:
    if not content_type.lower().startswith("multipart/form-data;"):
        raise ValueError("invalid content type")
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body)
    fields: dict[str, str] = {}
    audio: bytes | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if name == "audio" and part.get_content_type() == "audio/wav":
            audio = payload
        elif name in {"request_id", "robot_id", "language", "timestamp",
                      "context", "available_tools"}:
            fields[str(name)] = payload.decode("utf-8")
    if audio is None:
        raise ValueError("audio is required")
    return fields, audio


def _json_field(fields: dict[str, str], name: str, expected: type) -> Any:
    value = json.loads(fields[name])
    if not isinstance(value, expected):
        raise ValueError(f"{name} has the wrong shape")
    return value


def _notify_ready(version: str) -> None:
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
        channel.connect(address)
        channel.sendall(
            f"READY=1\nSTATUS=CUDA ASR and Needle are warm ({version})".encode()
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sparkie host voice gateway")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = GatewayConfig.load(args.config)
    gateway = VoiceGateway(config)
    server = GatewayServer((config.bind_address, config.port), gateway)
    print(json.dumps({"event": "gateway_started", "version": SOURCE_VERSION},
                     separators=(",", ":")))
    _notify_ready(SOURCE_VERSION)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        gateway.close()


if __name__ == "__main__":
    main()
