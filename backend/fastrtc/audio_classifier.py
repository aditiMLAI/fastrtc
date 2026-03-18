"""Two-stage audio classifier for distinguishing speech from noise.

Stage 1 uses acoustic heuristics (RMS, ZCR, spectral features).
Stage 2 uses ASR confirmation on chunks that passed Stage 1.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public enumerations & result types
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    {
        "wait",
        "stop",
        "hold",
        "pause",
        "no",
        "yes",
        "hey",
        "hello",
        "hi",
        "ok",
        "okay",
        "hold on",
        "wait a moment",
    }
)


class ClassificationResult(str, Enum):
    """Result of audio classification."""

    NOISE = "noise"
    REAL_SPEECH = "real_speech"
    POSSIBLE_SPEECH = "possible_speech"


# ---------------------------------------------------------------------------
# ASR backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ASRBackend(Protocol):
    """Protocol for ASR (automatic speech recognition) backends.

    Any object implementing this protocol can be injected into
    :class:`AudioClassifier` as the Stage 2 confirmation engine.
    """

    async def transcribe(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> str:
        """Transcribe an audio chunk.

        Args:
            audio: Mono float32 audio samples, values in [-1, 1].
            sample_rate: Sample rate of *audio* in Hz.

        Returns:
            Transcription string (empty string if nothing was recognised).
        """
        ...


# ---------------------------------------------------------------------------
# Conversation context
# ---------------------------------------------------------------------------


@dataclass
class _ContextEntry:
    """Single entry in the conversation history."""

    result: ClassificationResult
    timestamp: float
    transcript: str | None = None


@dataclass
class ConversationContext:
    """Rolling window of recent classification results.

    Maintains the last *max_history* classification events and exposes helper
    methods used by the classifier to adjust confidence scores.

    Attributes:
        max_history: Maximum number of entries to keep.
        history: Deque of recent classification entries.
    """

    max_history: int = 10
    history: deque[_ContextEntry] = field(default_factory=deque)

    def __post_init__(self) -> None:
        """Reinitialise history with the correct maxlen from *max_history*."""
        if not isinstance(self.history, deque) or self.history.maxlen != self.max_history:
            self.history = deque(self.history, maxlen=self.max_history)

    def record(
        self,
        result: ClassificationResult,
        transcript: str | None = None,
    ) -> None:
        """Record a new classification event.

        Args:
            result: The classification result.
            transcript: Optional ASR transcript associated with the result.
        """
        self.history.append(
            _ContextEntry(
                result=result,
                timestamp=time.monotonic(),
                transcript=transcript,
            )
        )

    def recent_speech_count(self, window_s: float = 5.0) -> int:
        """Count real-speech events within the last *window_s* seconds.

        Args:
            window_s: Time window in seconds.

        Returns:
            Number of REAL_SPEECH entries within the window.
        """
        cutoff = time.monotonic() - window_s
        return sum(
            1
            for e in self.history
            if e.result is ClassificationResult.REAL_SPEECH
            and e.timestamp >= cutoff
        )

    def confidence_boost(self) -> float:
        """Return an additive confidence boost based on recent speech history.

        After 3 or more recent real-speech events the boost reaches 0.2.

        Returns:
            Float in [0, 0.2].
        """
        count = self.recent_speech_count()
        return min(count * 0.07, 0.2)


# ---------------------------------------------------------------------------
# Heuristic helpers (pure functions, safe for executor)
# ---------------------------------------------------------------------------


def _rms(audio: NDArray[np.float32]) -> float:
    """Compute Root Mean Square energy of *audio*."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _zero_crossing_rate(audio: NDArray[np.float32]) -> float:
    """Compute normalised zero-crossing rate of *audio*."""
    if audio.size < 2:
        return 0.0
    signs = np.sign(audio)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings) / (audio.size - 1)


def _spectral_centroid(audio: NDArray[np.float32], sample_rate: int) -> float:
    """Compute spectral centroid of *audio* in Hz.

    Args:
        audio: Mono float32 audio array.
        sample_rate: Sample rate in Hz.

    Returns:
        Spectral centroid in Hz, or 0.0 if the spectrum is silent.
    """
    if audio.size == 0 or sample_rate <= 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64)))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    total = spectrum.sum()
    if total == 0.0:
        return 0.0
    return float(np.dot(freqs, spectrum) / total)


def _spectral_flatness(audio: NDArray[np.float32]) -> float:
    """Compute spectral flatness (Wiener entropy) of *audio*.

    Returns a value in [0, 1] where 1 = perfectly flat (white noise).

    Args:
        audio: Mono float32 audio array.

    Returns:
        Spectral flatness in [0, 1].
    """
    if audio.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64))) + 1e-10
    log_mean = np.mean(np.log(spectrum))
    arithmetic_mean = np.mean(spectrum)
    if arithmetic_mean == 0.0:
        return 0.0
    geometric_mean = np.exp(log_mean)
    return float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))


def _heuristic_classify(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    rms_threshold: float,
    zcr_max: float,
    spectral_centroid_min_hz: float,
    spectral_centroid_max_hz: float,
    spectral_flatness_max: float,
    min_speech_duration_s: float,
    energy_gate_threshold: float,
) -> ClassificationResult:
    """Run all acoustic heuristics and return a preliminary classification.

    Args:
        audio: Mono float32 audio samples.
        sample_rate: Sample rate in Hz.
        rms_threshold: Minimum RMS energy to be considered non-silent.
        zcr_max: Maximum zero-crossing rate for voiced speech.
        spectral_centroid_min_hz: Minimum centroid for speech.
        spectral_centroid_max_hz: Maximum centroid for speech.
        spectral_flatness_max: Maximum flatness before rejecting as noise.
        min_speech_duration_s: Minimum duration (s) for a real speech chunk.
        energy_gate_threshold: RMS gate for short chunks.

    Returns:
        NOISE if any heuristic rejects the chunk, POSSIBLE_SPEECH otherwise.
    """
    if audio.size == 0 or sample_rate <= 0:
        return ClassificationResult.NOISE

    rms = _rms(audio)
    if rms < rms_threshold:
        return ClassificationResult.NOISE

    duration_s = audio.size / sample_rate
    if duration_s < min_speech_duration_s and rms < energy_gate_threshold:
        return ClassificationResult.NOISE

    zcr = _zero_crossing_rate(audio)
    if zcr > zcr_max:
        return ClassificationResult.NOISE

    centroid = _spectral_centroid(audio, sample_rate)
    if centroid < spectral_centroid_min_hz or centroid > spectral_centroid_max_hz:
        return ClassificationResult.NOISE

    flatness = _spectral_flatness(audio)
    if flatness > spectral_flatness_max:
        return ClassificationResult.NOISE

    return ClassificationResult.POSSIBLE_SPEECH


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


class AudioClassifier:
    """Two-stage audio classifier combining heuristics with ASR confirmation.

    Stage 1 (synchronous, fast) applies acoustic heuristics to reject clear
    noise quickly.  Stage 2 (asynchronous) runs an ASR backend on chunks that
    passed Stage 1 to confirm whether real speech was detected.

    Context-awareness is implemented through :class:`ConversationContext`
    which tracks recent classification history and adjusts confidence scores.

    Args:
        asr_backend: An :class:`ASRBackend` instance for Stage 2 transcription.
            When ``None``, Stage 2 is skipped and POSSIBLE_SPEECH chunks are
            returned as REAL_SPEECH.
        rms_threshold: Minimum RMS energy (default 0.005).
        zcr_max: Maximum zero-crossing rate (default 0.35).
        spectral_centroid_min_hz: Min centroid Hz (default 80).
        spectral_centroid_max_hz: Max centroid Hz (default 4000).
        spectral_flatness_max: Max spectral flatness (default 0.8).
        min_speech_duration_s: Duration gate in seconds (default 0.05).
        energy_gate_threshold: RMS gate for short chunks (default 0.02).
        context: Optional external ConversationContext; one is created if
            not supplied.
    """

    def __init__(
        self,
        asr_backend: ASRBackend | None = None,
        *,
        rms_threshold: float = 0.005,
        zcr_max: float = 0.35,
        spectral_centroid_min_hz: float = 80.0,
        spectral_centroid_max_hz: float = 4000.0,
        spectral_flatness_max: float = 0.8,
        min_speech_duration_s: float = 0.05,
        energy_gate_threshold: float = 0.02,
        context: ConversationContext | None = None,
    ) -> None:
        self._asr = asr_backend
        self._rms_threshold = rms_threshold
        self._zcr_max = zcr_max
        self._centroid_min = spectral_centroid_min_hz
        self._centroid_max = spectral_centroid_max_hz
        self._flatness_max = spectral_flatness_max
        self._min_duration = min_speech_duration_s
        self._energy_gate = energy_gate_threshold
        self.context: ConversationContext = context or ConversationContext()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_heuristic(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        bot_speaking: bool,
    ) -> ClassificationResult:
        """Execute Stage 1 heuristics.

        When *bot_speaking* is True the energy threshold is doubled to reduce
        false positives from background chatter.

        Args:
            audio: Mono float32 audio samples.
            sample_rate: Sample rate in Hz.
            bot_speaking: Whether the bot is currently generating speech.

        Returns:
            NOISE or POSSIBLE_SPEECH.
        """
        rms_threshold = (
            self._rms_threshold * 2.0 if bot_speaking else self._rms_threshold
        )
        energy_gate = (
            self._energy_gate * 2.0 if bot_speaking else self._energy_gate
        )
        return _heuristic_classify(
            audio,
            sample_rate,
            rms_threshold=rms_threshold,
            zcr_max=self._zcr_max,
            spectral_centroid_min_hz=self._centroid_min,
            spectral_centroid_max_hz=self._centroid_max,
            spectral_flatness_max=self._flatness_max,
            min_speech_duration_s=self._min_duration,
            energy_gate_threshold=energy_gate,
        )

    async def _run_asr(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> tuple[ClassificationResult, str | None]:
        """Execute Stage 2 ASR confirmation.

        Args:
            audio: Mono float32 audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (ClassificationResult, transcript_or_None).
        """
        if self._asr is None:
            return ClassificationResult.REAL_SPEECH, None

        try:
            transcript = await self._asr.transcribe(audio, sample_rate)
        except Exception as exc:
            logger.warning(
                "ASR backend raised %s: %s – falling back to REAL_SPEECH",
                type(exc).__name__,
                exc,
            )
            return ClassificationResult.REAL_SPEECH, None

        stripped = transcript.strip().lower()
        if not stripped:
            return ClassificationResult.NOISE, transcript

        # Stopwords are never demoted to noise
        if stripped in STOPWORDS:
            return ClassificationResult.REAL_SPEECH, transcript

        return ClassificationResult.REAL_SPEECH, transcript

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        *,
        bot_speaking: bool = False,
    ) -> tuple[ClassificationResult, str | None, float]:
        """Classify an audio chunk.

        Runs Stage 1 heuristics (in a thread executor to avoid blocking the
        event loop) and, if Stage 1 passes, runs Stage 2 ASR confirmation.

        Args:
            audio: Mono float32 audio samples.
            sample_rate: Sample rate in Hz.
            bot_speaking: Whether the bot is currently generating speech.

        Returns:
            Tuple of (ClassificationResult, transcript_or_None, confidence).
        """
        loop = asyncio.get_event_loop()

        # Stage 1 – run heuristics off the event loop thread
        heuristic_result: ClassificationResult = await loop.run_in_executor(
            None,
            self._run_heuristic,
            audio,
            sample_rate,
            bot_speaking,
        )

        if heuristic_result is ClassificationResult.NOISE:
            confidence = max(0.0, 0.9 - self.context.confidence_boost())
            self.context.record(ClassificationResult.NOISE)
            return ClassificationResult.NOISE, None, confidence

        # Stage 2 – ASR confirmation
        result, transcript = await self._run_asr(audio, sample_rate)

        boost = self.context.confidence_boost()
        if result is ClassificationResult.REAL_SPEECH:
            confidence = min(1.0, 0.75 + boost)
        else:
            confidence = max(0.0, 0.85 - boost)

        self.context.record(result, transcript)
        return result, transcript, confidence
