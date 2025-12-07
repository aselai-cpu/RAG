"""
Chat Repository Interface - Defines the contract for chat storage.

This interface provides the abstraction for persisting and retrieving
chat sessions and messages.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.entities.chat import ChatSession, Message


class IChatRepository(ABC):
    """
    Interface for chat repository operations.

    This abstraction allows the domain layer to remain independent
    of how chat sessions are persisted.
    """

    @abstractmethod
    def save_session(self, session: ChatSession) -> None:
        """
        Save a chat session.

        Args:
            session: The chat session to save
        """
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve a chat session by ID.

        Args:
            session_id: The unique identifier of the session

        Returns:
            The chat session if found, None otherwise
        """
        pass

    @abstractmethod
    def get_all_sessions(self) -> List[ChatSession]:
        """
        Retrieve all chat sessions.

        Returns:
            List of all chat sessions
        """
        pass

    @abstractmethod
    def add_message(self, session_id: str, message: Message) -> None:
        """
        Add a message to a chat session.

        Args:
            session_id: The session to add the message to
            message: The message to add
        """
        pass
