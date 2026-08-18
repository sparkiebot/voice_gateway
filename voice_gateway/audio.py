"""Strict validation for completed robot utterances."""

from dataclasses import dataclass
from array import array
import os
from pathlib import Path
import sys
import tempfile
import wave


@dataclass(frozen=True)
class AudioInfo:
    duration_seconds: float
    bytes: int


def validate_wav(path: Path, max_seconds: float, max_bytes: int) -> AudioInfo:
    size = path.stat().st_size
    if size <= 44 or size > max_bytes:
        raise ValueError("invalid_audio")
    try:
        with wave.open(str(path), "rb") as source:
            valid = (source.getcomptype() == "NONE" and source.getnchannels() == 1
                     and source.getsampwidth() == 2 and source.getframerate() == 16_000
                     and source.getnframes() > 0)
            duration = source.getnframes() / source.getframerate()
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError("invalid_audio") from exc
    if not valid or duration > max_seconds:
        raise ValueError("invalid_audio")
    return AudioInfo(duration, size)


def trim_silence(
    path: Path,
    threshold_percent: float,
    leading_ms: int,
    trailing_ms: int,
    minimum_ms: int,
) -> tuple[Path, AudioInfo] | None:
    """Write a private trimmed PCM WAV, or return None when speech is unclear."""
    with wave.open(str(path), "rb") as source:
        sample_rate = source.getframerate()
        samples = array("h")
        samples.frombytes(source.readframes(source.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    window_samples = max(1, sample_rate // 100)  # 10 ms decisions.
    threshold = max(1, round(32767 * threshold_percent / 100.0))
    active_windows: list[int] = []
    for offset in range(0, len(samples), window_samples):
        window = samples[offset:offset + window_samples]
        if window and max(abs(sample) for sample in window) >= threshold:
            active_windows.append(offset)
    if not active_windows:
        return None
    padding_before = sample_rate * leading_ms // 1000
    padding_after = sample_rate * trailing_ms // 1000
    start = max(0, active_windows[0] - padding_before)
    end = min(len(samples), active_windows[-1] + window_samples + padding_after)
    if end - start < sample_rate * minimum_ms // 1000:
        return None
    descriptor, name = tempfile.mkstemp(prefix="sparkie-asr-", suffix=".wav")
    os.close(descriptor)
    output = Path(name)
    try:
        with wave.open(str(output), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(sample_rate)
            target.writeframes(samples[start:end].tobytes())
        os.chmod(output, 0o600)
        return output, AudioInfo((end - start) / sample_rate, output.stat().st_size)
    except Exception:
        output.unlink(missing_ok=True)
        raise
