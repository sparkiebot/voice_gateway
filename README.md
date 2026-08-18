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

## Updating production

From the production checkout, run:

```bash
cd /home/sparkie/sparkie/voice_gateway
./update.sh
```

The updater refuses dirty, detached, diverged, or non-`main` source states. It
fast-forwards from `origin/main`, preserves the ignored `runtime/` directory,
compiles and tests the new source, reinstalls the systemd unit, restarts the
service, and verifies that authenticated `/health` reports the new Git commit.
`VOICE_GATEWAY_UPDATE_REMOTE`, `VOICE_GATEWAY_UPDATE_BRANCH`, and
`VOICE_GATEWAY_SYSTEMD_SERVICE` may override their respective defaults.

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

## Adding tools and local responses

`tools.json` is the canonical tool catalogue. Add a tool there with a unique
snake_case `name`, a precise Italian description, and an object JSON Schema for
`parameters`. The robot request may enable a tool by name, but its supplied
description or schema is ignored: the gateway forwards the canonical catalogue
entry to Mind and validates Needle proposals against it. This keeps the contract
reviewable in Git and prevents a caller from broadening a tool at runtime.

Small talk is intentionally one `local_smalltalk` tool, not one tool per
response. Its enum maps to the intents in the response catalogue. To add a new
local answer, add an enum value and a matching catalogue entry. The data-driven
response form below makes its exact accepted utterances and answers live in one
place (old list-only entries remain supported for migration):

```json
{
  "wake_up": {
    "phrases": ["svegliati", "buongiorno"],
    "responses": ["Eccomi!", "Buongiorno, sono qui."]
  }
}
```

Use local responses only where an exact utterance is safe and complete. For
anything with variables, ambiguity, a side effect, or a response that depends
on context, define a normal tool and let Mind handle the final interaction.
The wake-word service consumes `Hey Sparkie`; do not add it to tool descriptions,
phrases, or fine-tuning queries.

## Needle fine-tuning

Start with the base model: its schema grammar, retrieval (only the best five
tools are reachable in a turn), and confidence gating are normally enough for a
small catalogue. Fine-tune only after recording an evaluation set that shows a
repeatable routing or argument-grounding problem.

Needle trains a LoRA adapter from JSONL examples. Each line carries the spoken
`query`, the tool schema(s) that were available, the expected `answers`, and a
short `reasoning` line that points every argument back to words in the query.
Include off-topic turns with `"answers": []` and deliberately ambiguous pairs
for similarly named tools. A few hundred clean examples help tool selection;
argument grounding generally needs thousands of varied examples.

```bash
needle finetune data/sparkie-tools.jsonl --epochs 15 --out checkpoints/sparkie_lora.pkl
needle build checkpoints/needle2.pkl --lora checkpoints/sparkie_lora.pkl --out models/sparkie-tools.cact
```

Point `needle_model_path` at the exported `.cact`. Important: Needle’s
confidence score is calibrated for base weights and is `None` for tuned weights,
so a tuned deployment must route using a separate explicit acceptance policy
(for example schema validation plus a conservative allowlist), rather than the
current confidence threshold alone. The gateway fails closed by default; after
reviewing the tool allowlist, set `allow_unscored_needle` to `true` in the
operator configuration to permit these schema-validated but unscored proposals.
