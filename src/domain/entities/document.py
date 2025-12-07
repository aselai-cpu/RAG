"""
Document Entity - Core domain model for documents in the RAG system.

This represents the domain concept of a document that can be ingested,
processed, and used for retrieval-augmented generation.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4


@dataclass
class Document:
    """
    Document entity representing a document in the system.

    Attributes:
        id: Unique identifier for the document
        content: The actual text content of the document
        source_type: Type of source (pdf, text, clipboard)
        file_name: Original file name if applicable
        metadata: Additional metadata about the document
        created_at: Timestamp when document was created
        chunk_count: Number of chunks this document was split into
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    source_type: str = "text"  # pdf, text, clipboard
    file_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    chunk_count: int = 0

    def __post_init__(self):
        """Validate document after initialization."""
        if not self.content:
            raise ValueError("Document content cannot be empty")
        if self.source_type not in ["pdf", "text", "clipboard"]:
            raise ValueError(f"Invalid source_type: {self.source_type}")

    def get_display_name(self) -> str:
        """Get a display name for the document."""
        if self.file_name:
            return self.file_name
        return f"{self.source_type.capitalize()} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    def to_dict(self) -> dict:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "source_type": self.source_type,
            "file_name": self.file_name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "chunk_count": self.chunk_count,
        }
