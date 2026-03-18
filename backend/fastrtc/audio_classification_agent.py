"""Intelligent interruption agent that classifies audio as noise or real speech.

The :class:`InterruptionAgent` wires together Stage 1 (heuristics) and Stage 2
(ASR) of :class:`~fastrtc.audio_classifier.AudioClassifier` and produces an
:class:`InterruptionDecision` indicating whether the response pipeline should
continue or pause.
"""

import asyncio
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .audio_classifier import (
    ASRBackend,
    AudioClassifier,
    ClassificationResult,
    ConversationContext,
)


@dataclass
class InterruptionDecision:
    """Decision produced by :class:`InterruptionAgent` for a single audio chunk.

    Attributes:
        action: ``"continue"`` to keep generating, ``"pause"`` to stop and wait.
        classification: The underlying :class:`ClassificationResult`.
        transcript: ASR transcript if available, else ``None``.
        confidence: Confidence score in [0, 1].
    """

    action: Literal["continue", "pause"]
    classification: ClassificationResult
    transcript: str | None
    confidence: float


class InterruptionAgent:
    """Async agent that determines whether an audio interruption is noise or speech.

    Uses :class:`~fastrtc.audio_classifier.AudioClassifier` internally.  Both
    the heuristic and ASR stages run without blocking the asyncio event loop.

    The agent supports the async context manager protocol; on ``__aexit__`` it
    awaits all in-flight ASR tasks before returning.

    Args:
        asr_backend: Optional :class:`~fastrtc.audio_classifier.ASRBackend`
            for Stage 2 ASR confirmation.  When ``None``, any chunk that passes
            heuristics is treated as real speech.
        classifier: Optional pre-constructed :class:`AudioClassifier`.  When
            supplied, *asr_backend* is ignored.
        context: Optional shared :class:`ConversationContext`.
    """

    def __init__(
        self,
        asr_backend: ASRBackend | None = None,
        *,
        classifier: AudioClassifier | None = None,
        context: ConversationContext | None = None,
    ) -> None:
        if classifier is not None:
            self._classifier = classifier
        else:
            self._classifier = AudioClassifier(
                asr_backend=asr_backend,
                context=context,
            )
        self._inflight: set[asyncio.Task[InterruptionDecision]] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def context(self) -> "ConversationContext":
        """Return the classifier's conversation context."""
        return self._classifier.context

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "InterruptionAgent":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Drain all in-flight ASR tasks before exiting."""
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)
        self._inflight.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_interruption(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int,
        *,
        bot_speaking: bool = False,
    ) -> InterruptionDecision:
        """Classify *audio_chunk* and decide whether to pause response generation.

        Args:
            audio_chunk: Mono float32 audio samples.
            sample_rate: Sample rate of *audio_chunk* in Hz.
            bot_speaking: ``True`` when the bot is currently generating speech;
                raises the noise threshold to reduce false positives.

        Returns:
            An :class:`InterruptionDecision` with ``action = "pause"`` for real
            speech and ``action = "continue"`` for noise.
        """
        result, transcript, confidence = await self._classifier.classify(
            audio_chunk,
            sample_rate,
            bot_speaking=bot_speaking,
        )

        action: Literal["continue", "pause"] = (
            "pause" if result is ClassificationResult.REAL_SPEECH else "continue"
        )

        return InterruptionDecision(
            action=action,
            classification=result,
            transcript=transcript,
            confidence=confidence,
        )
