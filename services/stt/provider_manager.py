"""Provider manager — registry for STT providers.

Maps display names (shown in the UI dropdown) to provider instances.
The UI never uses if/else to select a provider; it passes the dropdown
value straight to :meth:`get_provider`.
"""

from services.stt.base import BaseSTTProvider, STTError
from utils.logger import get_logger

logger = get_logger(__name__)


class ProviderManager:
    """Registry that maps display names to :class:`BaseSTTProvider` instances.

    Usage::

        manager = ProviderManager()
        manager.register("Deepgram Nova-3", deepgram_provider)
        provider = manager.get_provider("Deepgram Nova-3")
        result = provider.transcribe(audio, language, model)
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseSTTProvider] = {}

    def register(self, display_name: str, provider: BaseSTTProvider) -> None:
        """Register a provider under a display name.

        Args:
            display_name: Name shown in the UI dropdown.
            provider: An instance of :class:`BaseSTTProvider`.
        """
        self._providers[display_name] = provider
        logger.info("Registered STT provider: %s", display_name)

    def get_provider(self, display_name: str) -> BaseSTTProvider:
        """Look up a provider by its display name.

        Args:
            display_name: Exact name as registered / shown in dropdown.

        Returns:
            The corresponding provider instance.

        Raises:
            STTError: If no provider is registered under that name.
        """
        provider = self._providers.get(display_name)
        if provider is None:
            available = ", ".join(self._providers.keys()) or "(none)"
            raise STTError(
                f"Unknown STT provider: '{display_name}'. "
                f"Available: {available}"
            )
        return provider

    def get_available_providers(self) -> list[str]:
        """Return the display names of all registered providers.

        Returns:
            List of display name strings (ordered by registration).
        """
        return list(self._providers.keys())
