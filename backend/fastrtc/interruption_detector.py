"""Thin orchestration layer on top of :class:`InterruptionAgent`.

:class:`InterruptionDetector` exposes an :class:`asyncio.Event`
(``speech_detected_event``) that is set when real speech is detected and
cleared when noise is detected.  The response pipeline can ``await`` the event
or call :meth:`wait_for_speech_end` to block until the user finishes speaking.
"""

import asyncio

import numpy as np
from numpy.typing import NDArray

from .audio_classification_agent import InterruptionAgent, InterruptionDecision
from .audio_classifier import ASRBackend, ClassificationResult, ConversationContext


class InterruptionDetector:
    """Orchestration layer that bridges audio chunks with response-pipeline control.

    Wraps an :class:`InterruptionAgent` and exposes an asyncio event that the
    response pipeline can await to know when the user is speaking.

    Args:
        agent: Optional pre-constructed :class:`InterruptionAgent`.  When
            ``None`` a default agent (no ASR backend) is created.
        asr_backend: Optional :class:`~fastrtc.audio_classifier.ASRBackend`
            forwarded to the :class:`InterruptionAgent` if *agent* is ``None``.
        context: Optional shared :class:`~fastrtc.audio_classifier.ConversationContext`.
    """

    def __init__(
        self,
        agent: InterruptionAgent | None = None,
        *,
        asr_backend: ASRBackend | None = None,
        context: ConversationContext | None = None,
    ) -> None:
        self._agent = agent or InterruptionAgent(
            asr_backend=asr_backend,
            context=context,
        )
        self.speech_detected_event: asyncio.Event = asyncio.Event()

    async def process(
        self,
        chunk: NDArray[np.float32],
        sample_rate: int,
        bot_speaking: bool = False,
    ) -> InterruptionDecision:
        """Process a single audio chunk and update the speech detection event.

        If the agent decides the chunk contains real speech the
        ``speech_detected_event`` is set; otherwise it is cleared.

        Args:
            chunk: Mono float32 audio samples.
            sample_rate: Sample rate of *chunk* in Hz.
            bot_speaking: Whether the bot is currently generating speech.

        Returns:
            The :class:`InterruptionDecision` produced by the agent.
        """
        decision = await self._agent.handle_interruption(
            chunk,
            sample_rate,
            bot_speaking=bot_speaking,
        )

        if decision.action == "pause":
            self.speech_detected_event.set()
        else:
            self.speech_detected_event.clear()

        return decision

    async def wait_for_speech_end(
        self,
        timeout: float | None = None,
    ) -> bool:
        """Wait until speech is detected or *timeout* seconds elapse.

        Args:
            timeout: Maximum time to wait in seconds.  ``None`` means wait
                indefinitely.

        Returns:
            ``True`` if speech was detected within the timeout, ``False`` if
            the timeout elapsed before speech was detected.
        """
        try:
            await asyncio.wait_for(
                self.speech_detected_event.wait(),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def last_classification(self) -> ClassificationResult | None:
        """Return the last classification result from the agent's context, if any."""
        history = self._agent.context.history
        if not history:
            return None
        return history[-1].result
