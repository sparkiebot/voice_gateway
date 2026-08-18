# Voice Gateway

This host-only service is the production boundary between the ROS 2 Distrobox
and Jetson CUDA voice inference. It eagerly initializes NeMo-Speech Q4 and
Needle, performs exactly one inference at a time, and never executes a tool.

The authenticated interface is deliberately small:

- `GET /health` returns readiness only after both models are warm.
- `POST /v1/voice-requests` accepts the completed 16 kHz mono PCM16 WAV and the
  existing multipart metadata contract.
- `429 busy` is returned when the single inference slot and configured bounded
  waiting slot are occupied.

Any uncertain, conversational, malformed, sensitive, unavailable, or failed
local route sends the NeMo transcript, context, and request ID as JSON to Mind
`POST /v1/text-requests`. Audio never leaves the gateway. Tool calls from a
fallback response are discarded in this milestone. Responses are cached by
request ID for safe transport retries.

Use `config.example.json` as the operator-owned `/etc/sparkie/voice-gateway.json`
and keep the shared secret out of Git. On Sparkie, copy
`config/voice-gateway.env.example` to the ignored `runtime/secret.env`, replace
the placeholder, and set mode `0600`. Install the unit with
`bin/install-service` after provisioning the runtime and reviewing the config.

Runtime-only files live in the ignored `runtime/` directory. The production
service needs `venv/`, the Q4 model under
`models/`, and the NeMo/ggml shared libraries under `lib/`.

## Development

The gateway itself has no mandatory Python package dependencies. The production
runtime additionally provides Needle and the native NeMo-Speech libraries.

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile voice_gateway/*.py tests/*.py
```

The default systemd unit assumes this repository is checked out at
`/home/sparkie/sparkie/voice_gateway`. Paths remain configurable in
`/etc/sparkie/voice-gateway.json`.

Set `log_transcripts` in the operator configuration to emit a structured
`voice_transcript` journal entry after ASR. It defaults to false because spoken
content may be sensitive; audio and context are never logged. Each entry also
contains `asr_inference_ms`, measured around the NeMo transcription call only.

Silence trimming is enabled by default. A private temporary WAV is trimmed with
10 ms PCM amplitude decisions and passed only to NeMo; it is deleted immediately
after ASR. The original robot WAV is never modified and is never sent to Mind.
`trim_threshold_percent`, `trim_leading_ms`,
`trim_trailing_ms`, and `trim_minimum_ms` are operator-configurable.

Harmless Italian small talk can be answered locally without Mind. Needle selects
one gateway-only intent tool, the gateway independently validates the normalized
phrase against that intent, then selects a random response from
`responses.it.json`. The same response is never selected twice consecutively for
one intent. Local intent tools are never advertised to Mind or ROS and cannot
execute physical actions. Unknown, ambiguous, mismatched, or low-confidence
phrases continue through the normal Mind text fallback.
