"""
Chat and Message Entities - Core domain models for chat interactions.

These represent the domain concepts of conversations and individual
messages in the RAG system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import uuid4


@dataclass
class Message:
    """
    Message entity representing a single message in a chat.

    Attributes:
        id: Unique identifier for the message
        role: Role of the message sender (user, assistant, system)
        content: The message content
        created_at: Timestamp when message was created
        sources: List of document IDs used to generate this response
        metadata: Additional metadata about the message
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    role: str = "user"  # user, assistant, system
    content: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate message after initialization."""
        if not self.content:
            raise ValueError("Message content cannot be empty")
        if self.role not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {self.role}")

    def to_dict(self) -> dict:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "sources": self.sources,
            "metadata": self.metadata,
        }


@dataclass
class ChatSession:
    """
    ChatSession entity representing a conversation session.

    Attributes:
        id: Unique identifier for the chat session
        messages: List of messages in the session
        created_at: Timestamp when session was created
        updated_at: Timestamp when session was last updated
        metadata: Additional metadata about the session
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the chat session."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_history(self, limit: Optional[int] = None) -> List[Message]:
        """Get chat history, optionally limited to recent messages."""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def to_dict(self) -> dict:
        """Convert chat session to dictionary representation."""
        return {
            "id": self.id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }
