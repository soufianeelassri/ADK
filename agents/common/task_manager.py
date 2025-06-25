from abc import ABC, abstractmethod
from typing import AsyncIterable, Dict, Any

class AgentWithTaskManager(ABC):
    """Abstract base class for agents that can be managed by the A2A server."""
    @abstractmethod
    def get_processing_message(self) -> str:
        """Returns a message to show while the task is processing."""
        raise NotImplementedError()

    @abstractmethod
    async def run(self, user_id: str, message: str) -> AsyncIterable[Dict[str, Any]]:
        """Runs the agent's logic and yields updates."""
        if False:
            yield