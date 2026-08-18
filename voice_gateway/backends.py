"""Thin ASR/router adapters; no tool execution."""

from array import array
import ctypes
from pathlib import Path
from typing import Any
import sys
import wave


class _BackendConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("gpu", ctypes.c_int32)]


class _ModelConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("path", ctypes.c_char_p),
                ("name", ctypes.c_char_p)]


class _RecognizerConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("backend", ctypes.c_void_p),
                ("model", ctypes.c_void_p), ("streaming", ctypes.c_void_p),
                ("decoder", ctypes.c_void_p), ("vad", ctypes.c_void_p),
                ("endpointing", ctypes.c_void_p), ("postproc", ctypes.c_void_p),
                ("diar", ctypes.c_void_p), ("batching", ctypes.c_void_p)]


class _RecognitionOptions(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t), ("request_id", ctypes.c_char_p),
        ("language_code", ctypes.c_char_p), ("interim_results", ctypes.c_bool),
        ("enable_word_time_offsets", ctypes.c_bool),
        ("enable_automatic_punctuation", ctypes.c_bool),
        ("verbatim_transcripts", ctypes.c_bool), ("profanity_filter", ctypes.c_bool),
        ("stop_history_eou_ms", ctypes.c_int32), ("speech_contexts", ctypes.c_void_p),
        ("speech_context_count", ctypes.c_size_t), ("max_alternatives", ctypes.c_int32),
        ("enable_speaker_diarization", ctypes.c_bool),
        ("max_speaker_count", ctypes.c_int32),
    ]


class NemoSpeechCuda:
    """Persistent NeMo-Speech.cpp CUDA recognizer accessed through its stable C ABI."""

    def __init__(self, model_path: str, library_path: str, language: str = "it",
                 gpu: int = 0, warmup_seconds: float = 2.0) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.library_path = str(Path(library_path).expanduser().resolve())
        self.language = language
        self.gpu = gpu
        self.warmup_seconds = max(0.0, warmup_seconds)
        self._lib: Any | None = None
        self._recognizer = ctypes.c_void_p()

    def _error(self, operation: str, status: int) -> RuntimeError:
        raw = self._lib.nemo_speech_asr_last_error()
        detail = raw.decode("utf-8", "replace") if raw else "unknown error"
        return RuntimeError(f"NeMo-Speech {operation} failed ({status}): {detail}")

    def initialize(self) -> None:
        if self._recognizer.value is not None:
            return
        if not Path(self.model_path).is_file():
            raise RuntimeError(f"NeMo-Speech model not found: {self.model_path}")
        if not Path(self.library_path).is_file():
            raise RuntimeError(f"NeMo-Speech library not found: {self.library_path}")
        lib = ctypes.CDLL(self.library_path)
        lib.nemo_speech_asr_create.argtypes = [ctypes.POINTER(_RecognizerConfig),
                                               ctypes.POINTER(ctypes.c_void_p)]
        lib.nemo_speech_asr_create.restype = ctypes.c_int
        lib.nemo_speech_asr_destroy.argtypes = [ctypes.c_void_p]
        lib.nemo_speech_asr_recognize_f32.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_RecognitionOptions),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.nemo_speech_asr_recognize_f32.restype = ctypes.c_int
        lib.nemo_speech_asr_result_transcript.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_size_t]
        lib.nemo_speech_asr_result_transcript.restype = ctypes.c_char_p
        lib.nemo_speech_asr_result_confidence.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_size_t]
        lib.nemo_speech_asr_result_confidence.restype = ctypes.c_float
        lib.nemo_speech_asr_result_destroy.argtypes = [ctypes.c_void_p]
        lib.nemo_speech_asr_last_error.restype = ctypes.c_char_p

        backend = _BackendConfig(ctypes.sizeof(_BackendConfig), self.gpu)
        model = _ModelConfig(ctypes.sizeof(_ModelConfig), self.model_path.encode(), None)
        config = _RecognizerConfig()
        config.size = ctypes.sizeof(_RecognizerConfig)
        config.backend = ctypes.cast(ctypes.pointer(backend), ctypes.c_void_p)
        config.model = ctypes.cast(ctypes.pointer(model), ctypes.c_void_p)
        recognizer = ctypes.c_void_p()
        self._lib = lib
        status = lib.nemo_speech_asr_create(ctypes.byref(config), ctypes.byref(recognizer))
        if status != 0:
            raise self._error("initialization", status)
        self._recognizer = recognizer
        if self.warmup_seconds:
            # A non-empty silent request materializes lazy CUDA graphs during startup.
            count = max(1, int(16000 * self.warmup_seconds))
            silence = (ctypes.c_float * count)()
            options = self._options()
            result = ctypes.c_void_p()
            status = lib.nemo_speech_asr_recognize_f32(
                recognizer, ctypes.byref(options), silence, count, 16000,
                ctypes.byref(result),
            )
            if status != 0:
                self.close()
                raise self._error("warm-up", status)
            if result.value is not None:
                lib.nemo_speech_asr_result_destroy(result)

    def _options(self) -> _RecognitionOptions:
        options = _RecognitionOptions()
        options.size = ctypes.sizeof(_RecognitionOptions)
        options.language_code = self.language.encode()
        options.max_alternatives = 1
        return options

    @staticmethod
    def _read_pcm16_mono(wav_path: Path) -> tuple[array, int]:
        with wave.open(str(wav_path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise RuntimeError("NeMo-Speech POC requires uncompressed 16-bit PCM WAV")
            if channels not in (1, 2):
                raise RuntimeError("NeMo-Speech POC accepts mono or stereo WAV")
            pcm = array("h")
            pcm.frombytes(source.readframes(source.getnframes()))
        if sys.byteorder != "little":
            pcm.byteswap()
        if channels == 2:
            pcm = array("h", ((int(pcm[i]) + int(pcm[i + 1])) // 2
                              for i in range(0, len(pcm), 2)))
        return array("f", (sample / 32768.0 for sample in pcm)), sample_rate

    def transcribe(self, wav_path: Path) -> dict[str, Any]:
        if self._recognizer.value is None:
            raise RuntimeError("NemoSpeechCuda.initialize() must run before inference")
        samples, sample_rate = self._read_pcm16_mono(wav_path)
        if not samples:
            raise RuntimeError("NeMo-Speech received empty audio")
        buffer = (ctypes.c_float * len(samples)).from_buffer(samples)
        options = self._options()
        result = ctypes.c_void_p()
        status = self._lib.nemo_speech_asr_recognize_f32(
            self._recognizer, ctypes.byref(options), buffer, len(samples), sample_rate,
            ctypes.byref(result),
        )
        if status != 0:
            raise self._error("inference", status)
        if result.value is None:
            raise RuntimeError("NeMo-Speech returned no result")
        try:
            raw_text = self._lib.nemo_speech_asr_result_transcript(result, 0)
            text = raw_text.decode("utf-8", "replace").strip() if raw_text else ""
            confidence = float(self._lib.nemo_speech_asr_result_confidence(result, 0))
            return {"text": text, "confidence": confidence,
                    "metrics": {"backend": "nemo_speech_cuda", "gpu": self.gpu,
                                "sample_rate": sample_rate}}
        finally:
            self._lib.nemo_speech_asr_result_destroy(result)

    def close(self) -> None:
        if self._lib is not None and self._recognizer.value is not None:
            self._lib.nemo_speech_asr_destroy(self._recognizer)
            self._recognizer = ctypes.c_void_p()


class NeedleRouter:
    def __init__(self, tools: list[dict[str, Any]], model_path: str = "",
                 system: str = "") -> None:
        try:
            import needle  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install the cactus-needle package") from exc
        kwargs: dict[str, Any] = {"tools": tools}
        if model_path:
            kwargs["weights"] = model_path
        if system:
            kwargs["system"] = system
        self.agent = needle.Needle(**kwargs)

    def complete(self, transcript: str) -> dict[str, Any]:
        # Deliberately use complete(), never run(); run() can execute functions.
        # Each robot utterance is independent; retaining Needle history caused
        # later requests to be interpreted as continuations of earlier ones.
        self.agent.reset()
        result = self.agent.complete(transcript)
        if not isinstance(result, dict):
            raise RuntimeError("Needle returned a non-object result")
        return result
