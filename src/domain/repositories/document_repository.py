"""
Document Repository Interface - Defines the contract for document storage.

This is the Anti-Corruption Layer between the domain and infrastructure.
The domain doesn't care HOW documents are stored, only THAT they can be stored.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.entities.document import Document


class IDocumentRepository(ABC):
    """
    Interface for document repository operations.

    This abstraction allows the domain layer to remain independent
    of infrastructure concerns like vector databases.
    """

    @abstractmethod
    def save(self, document: Document) -> None:
        """
        Save a document to the repository.

        Args:
            document: The document to save
        """
        pass

    @abstractmethod
    def get_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.

        Args:
            document_id: The unique identifier of the document

        Returns:
            The document if found, None otherwise
        """
        pass

    @abstractmethod
    def get_all(self) -> List[Document]:
        """
        Retrieve all documents.

        Returns:
            List of all documents in the repository
        """
        pass

    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """
        Delete a document from the repository.

        Args:
            document_id: The unique identifier of the document

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    def search_similar(self, query: str, top_k: int = 5) -> List[tuple[Document, float]]:
        """
        Search for documents similar to the query.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of tuples containing (document, similarity_score)
        """
        pass
