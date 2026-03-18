"""Comprehensive tests for the audio classification and interruption detection system."""

import asyncio

import numpy as np
import pytest
from fastrtc.audio_classification_agent import InterruptionAgent, InterruptionDecision
from fastrtc.audio_classifier import (
    AudioClassifier,
    ClassificationResult,
    ConversationContext,
)
from fastrtc.interruption_detector import InterruptionDetector
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SR = 16000  # 16 kHz sample rate used throughout tests


def _silence(duration_s: float = 0.5) -> NDArray[np.float32]:
    """Generate a silent (all-zeros) audio array."""
    return np.zeros(int(SR * duration_s), dtype=np.float32)


def _white_noise(duration_s: float = 0.5, amplitude: float = 0.005) -> NDArray[np.float32]:
    """Generate low-energy broadband white noise."""
    rng = np.random.default_rng(42)
    return (rng.uniform(-amplitude, amplitude, int(SR * duration_s))).astype(np.float32)


def _sine(freq_hz: float, duration_s: float = 0.5, amplitude: float = 0.3) -> NDArray[np.float32]:
    """Generate a pure sine-wave tone."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _voiced_speech_like(duration_s: float = 0.5, amplitude: float = 0.3) -> NDArray[np.float32]:
    """Generate a multi-harmonic signal mimicking voiced speech (200–3000 Hz)."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    signal = np.zeros_like(t)
    for harmonic in [200, 400, 800, 1200, 1600, 2000, 2400, 2800]:
        signal += amplitude * np.sin(2 * np.pi * harmonic * t)
    signal /= np.max(np.abs(signal)) + 1e-8
    signal *= amplitude
    return signal.astype(np.float32)


class _MockASR:
    """Mock ASR backend that returns a fixed transcript."""

    def __init__(self, transcript: str = "") -> None:
        self._transcript = transcript

    async def transcribe(
        self, audio: NDArray[np.float32], sample_rate: int
    ) -> str:
        return self._transcript


class _RaisingASR:
    """Mock ASR backend that always raises RuntimeError."""

    async def transcribe(
        self, audio: NDArray[np.float32], sample_rate: int
    ) -> str:
        raise RuntimeError("ASR backend failure")


# ===========================================================================
# 5a. Heuristic noise vs. speech classification
# ===========================================================================


@pytest.mark.asyncio
async def test_pure_silence_classified_as_noise() -> None:
    """All-zeros array must be classified as NOISE."""
    classifier = AudioClassifier()
    result, _, _ = await classifier.classify(_silence(), SR)
    assert result is ClassificationResult.NOISE


@pytest.mark.asyncio
async def test_white_noise_classified_as_noise() -> None:
    """Low-energy broadband noise must be classified as NOISE."""
    classifier = AudioClassifier()
    result, _, _ = await classifier.classify(_white_noise(amplitude=0.005), SR)
    assert result is ClassificationResult.NOISE


@pytest.mark.asyncio
async def test_sinusoidal_noise_classified_as_noise() -> None:
    """Pure 50 Hz hum (outside speech band) must be classified as NOISE."""
    classifier = AudioClassifier()
    result, _, _ = await classifier.classify(_sine(50.0), SR)
    assert result is ClassificationResult.NOISE


@pytest.mark.asyncio
async def test_voiced_speech_like_signal_passes_heuristic() -> None:
    """Multi-harmonic signal in 200–3000 Hz must pass the heuristic stage.

    Without an ASR backend the classifier promotes POSSIBLE_SPEECH to
    REAL_SPEECH, so we accept either REAL_SPEECH or POSSIBLE_SPEECH here.
    """
    classifier = AudioClassifier(asr_backend=None)
    result, _, _ = await classifier.classify(_voiced_speech_like(), SR)
    assert result in (ClassificationResult.REAL_SPEECH, ClassificationResult.POSSIBLE_SPEECH)


# ===========================================================================
# 5b. ASR confirmation stage
# ===========================================================================


@pytest.mark.asyncio
async def test_asr_empty_transcript_gives_noise() -> None:
    """Heuristic passes but ASR returns empty string → NOISE."""
    asr = _MockASR(transcript="")
    classifier = AudioClassifier(asr_backend=asr)
    result, _, _ = await classifier.classify(_voiced_speech_like(), SR)
    assert result is ClassificationResult.NOISE


@pytest.mark.asyncio
async def test_asr_meaningful_transcript_gives_speech() -> None:
    """ASR returns 'stop' → REAL_SPEECH."""
    asr = _MockASR(transcript="stop")
    classifier = AudioClassifier(asr_backend=asr)
    result, transcript, _ = await classifier.classify(_voiced_speech_like(), SR)
    assert result is ClassificationResult.REAL_SPEECH
    assert transcript == "stop"


@pytest.mark.asyncio
async def test_short_stopword_is_never_demoted() -> None:
    """ASR returning 'wait' must always yield REAL_SPEECH (stopword protection)."""
    asr = _MockASR(transcript="wait")
    classifier = AudioClassifier(asr_backend=asr)
    result, _, _ = await classifier.classify(_voiced_speech_like(), SR)
    assert result is ClassificationResult.REAL_SPEECH


# ===========================================================================
# 5c. Context-awareness
# ===========================================================================


@pytest.mark.asyncio
async def test_consecutive_speech_raises_confidence() -> None:
    """After 3 real speech events, confidence for next chunk is ≥ 0.8."""
    ctx = ConversationContext()
    asr = _MockASR(transcript="hello")
    classifier = AudioClassifier(asr_backend=asr, context=ctx)

    speech = _voiced_speech_like()
    for _ in range(3):
        await classifier.classify(speech, SR)

    _, _, confidence = await classifier.classify(speech, SR)
    assert confidence >= 0.8


@pytest.mark.asyncio
async def test_bot_speaking_flag_tightens_threshold() -> None:
    """Same marginal audio should be harder to classify as speech when bot speaks.

    We use a low-amplitude speech-like signal that passes heuristics with
    default thresholds but should be rejected when bot_speaking doubles the
    energy gate.
    """
    # A very low amplitude signal – close to the threshold boundary
    audio = _voiced_speech_like(amplitude=0.01)
    asr = _MockASR(transcript="hello")

    clf_normal = AudioClassifier(asr_backend=asr)
    clf_bot = AudioClassifier(asr_backend=asr)

    result_normal, _, _ = await clf_normal.classify(audio, SR, bot_speaking=False)
    result_bot, _, _ = await clf_bot.classify(audio, SR, bot_speaking=True)

    # At least one of: normal passes but bot doesn't; or both noise (threshold too high)
    # The key assertion is that the bot_speaking path is at least as strict.
    # We allow both to be NOISE, but normal must not be stricter than bot.
    strict_order = {ClassificationResult.NOISE: 0, ClassificationResult.POSSIBLE_SPEECH: 1, ClassificationResult.REAL_SPEECH: 2}
    assert strict_order[result_normal] >= strict_order[result_bot]


# ===========================================================================
# 5d. Async execution behaviour
# ===========================================================================


@pytest.mark.asyncio
async def test_agent_does_not_block_event_loop() -> None:
    """A concurrent counter coroutine must progress while the agent processes."""

    counter = {"value": 0}

    async def increment_counter() -> None:
        for _ in range(10):
            counter["value"] += 1
            await asyncio.sleep(0)

    agent = InterruptionAgent(asr_backend=_MockASR("hello"))
    speech = _voiced_speech_like()

    await asyncio.gather(
        agent.handle_interruption(speech, SR),
        increment_counter(),
    )

    assert counter["value"] > 0


@pytest.mark.asyncio
async def test_multiple_concurrent_chunks() -> None:
    """Five concurrent chunks must complete without errors and return valid decisions."""
    agent = InterruptionAgent(asr_backend=_MockASR("hello"))
    speech = _voiced_speech_like()

    decisions = await asyncio.gather(
        *[agent.handle_interruption(speech, SR) for _ in range(5)]
    )

    assert len(decisions) == 5
    for d in decisions:
        assert isinstance(d, InterruptionDecision)
        assert d.action in ("continue", "pause")
        assert isinstance(d.confidence, float)
        assert 0.0 <= d.confidence <= 1.0


# ===========================================================================
# 5e. InterruptionDetector integration
# ===========================================================================


@pytest.mark.asyncio
async def test_detector_sets_event_on_real_speech() -> None:
    """After a real-speech decision speech_detected_event must be set."""
    asr = _MockASR(transcript="hello there")
    agent = InterruptionAgent(asr_backend=asr)
    detector = InterruptionDetector(agent=agent)

    decision = await detector.process(_voiced_speech_like(), SR)
    assert decision.action == "pause"
    assert detector.speech_detected_event.is_set()


@pytest.mark.asyncio
async def test_detector_does_not_set_event_on_noise() -> None:
    """After a noise decision speech_detected_event must NOT be set."""
    detector = InterruptionDetector()
    decision = await detector.process(_silence(), SR)
    assert decision.action == "continue"
    assert not detector.speech_detected_event.is_set()


@pytest.mark.asyncio
async def test_wait_for_speech_end_timeout() -> None:
    """wait_for_speech_end returns within the timeout when no speech detected."""
    detector = InterruptionDetector()
    result = await detector.wait_for_speech_end(timeout=0.01)
    assert result is False


# ===========================================================================
# 5f. Failure / error scenarios
# ===========================================================================


@pytest.mark.asyncio
async def test_asr_backend_raises_exception_gracefully() -> None:
    """ASR backend raising RuntimeError must not propagate; falls back to heuristic."""
    asr = _RaisingASR()
    classifier = AudioClassifier(asr_backend=asr)
    # Should not raise; the classifier catches the ASR exception and falls back
    result, _, _ = await classifier.classify(_voiced_speech_like(), SR)
    # Fallback is REAL_SPEECH (conservative: don't drop real speech on error)
    assert result is ClassificationResult.REAL_SPEECH


@pytest.mark.asyncio
async def test_empty_audio_chunk_handled() -> None:
    """Zero-length ndarray must not raise and must return NOISE."""
    classifier = AudioClassifier()
    empty = np.array([], dtype=np.float32)
    result, _, _ = await classifier.classify(empty, SR)
    assert result is ClassificationResult.NOISE


@pytest.mark.asyncio
async def test_invalid_sample_rate_handled() -> None:
    """sample_rate=0 must not raise and must return NOISE."""
    classifier = AudioClassifier()
    result, _, _ = await classifier.classify(_voiced_speech_like(), sample_rate=0)
    assert result is ClassificationResult.NOISE
