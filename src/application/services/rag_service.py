"""
RAG Service - Application layer orchestrating the RAG workflow.

This service coordinates between the domain, infrastructure, and
implements the core RAG (Retrieval-Augmented Generation) logic.
"""
from typing import List, Optional, Tuple
from src.domain.entities.document import Document
from src.domain.entities.chat import Message, ChatSession
from src.domain.repositories.document_repository import IDocumentRepository
from src.infrastructure.llm.openai_service import OpenAIService


class RAGService:
    """
    Service orchestrating the Retrieval-Augmented Generation workflow.

    This implements the classic RAG pattern:
    1. User asks a question
    2. Retrieve relevant documents from vector store
    3. Augment the prompt with retrieved context
    4. Generate response using LLM
    """

    def __init__(
        self,
        document_repository: IDocumentRepository,
        llm_service: OpenAIService,
        top_k_retrieval: int = 5,
        similarity_threshold: float = 0.4,
    ):
        """
        Initialize the RAG service.

        Args:
            document_repository: Repository for document storage and retrieval
            llm_service: Service for LLM interactions
            top_k_retrieval: Number of top documents to retrieve
            similarity_threshold: Minimum similarity score for retrieval
        """
        self.document_repository = document_repository
        self.llm_service = llm_service
        self.top_k_retrieval = top_k_retrieval
        self.similarity_threshold = similarity_threshold

    def query(
        self,
        question: str,
        chat_history: Optional[List[Message]] = None,
    ) -> Tuple[str, List[str], List[dict]]:
        """
        Process a RAG query.

        This implements the core RAG workflow:
        1. Retrieve relevant documents
        2. Build context from retrieved documents
        3. Generate response with context
        4. Return response, source document IDs, and detailed source information

        Args:
            question: The user's question
            chat_history: Optional chat history for context

        Returns:
            Tuple of (response_text, list_of_source_document_ids, list_of_source_details)
            Source details include: document_id, chunk_index, relevance_score, file_name
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.document_repository.search_similar(
            query=question, top_k=self.top_k_retrieval
        )

        # Filter by similarity threshold
        relevant_docs = [
            (doc, score)
            for doc, score in retrieved_docs
            if score >= self.similarity_threshold
        ]

        # Step 2: Build context from retrieved documents
        context = self._build_context(relevant_docs)

        # Step 3: Build messages for LLM
        messages = []

        # Add recent chat history if provided (last 5 messages)
        if chat_history:
            recent_history = chat_history[-5:]
            for msg in recent_history:
                messages.append({"role": msg.role, "content": msg.content})

        # Add current question
        messages.append({"role": "user", "content": question})

        # Step 4: Generate response
        response = self.llm_service.generate_response(
            messages=messages, context=context if relevant_docs else None
        )

        # Extract source information with chunk details
        sources = []
        for doc, score in relevant_docs:
            chunk_index = doc.metadata.get("chunk_index", 0)
            sources.append({
                "document_id": doc.id,
                "chunk_index": chunk_index,
                "relevance_score": score,
                "file_name": doc.file_name
            })
        
        # Also return unique document IDs for backward compatibility
        source_ids = list(set(doc.id for doc, _ in relevant_docs))

        return response, source_ids, sources

    def query_stream(
        self,
        question: str,
        chat_history: Optional[List[Message]] = None,
    ):
        """
        Process a RAG query with streaming response.

        Similar to query() but yields response chunks as they're generated.

        Args:
            question: The user's question
            chat_history: Optional chat history for context

        Yields:
            Tuples of (response_chunk, source_ids, source_details)
            The source_ids and source_details will be sent with the first chunk
            Source details include: document_id, chunk_index, relevance_score, file_name
        """
        # Retrieve relevant documents
        retrieved_docs = self.document_repository.search_similar(
            query=question, top_k=self.top_k_retrieval
        )

        # Filter by similarity threshold
        relevant_docs = [
            (doc, score)
            for doc, score in retrieved_docs
            if score >= self.similarity_threshold
        ]

        # Build context
        context = self._build_context(relevant_docs)

        # Build messages
        messages = []
        if chat_history:
            recent_history = chat_history[-5:]
            for msg in recent_history:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": question})

        # Extract source information with chunk details
        sources = []
        for doc, score in relevant_docs:
            chunk_index = doc.metadata.get("chunk_index", 0)
            sources.append({
                "document_id": doc.id,
                "chunk_index": chunk_index,
                "relevance_score": score,
                "file_name": doc.file_name
            })
        
        # Also return unique document IDs for backward compatibility
        source_ids = list(set(doc.id for doc, _ in relevant_docs))

        # Generate streaming response
        first_chunk = True
        for chunk in self.llm_service.generate_response_stream(
            messages=messages, context=context if relevant_docs else None
        ):
            if first_chunk:
                yield chunk, source_ids, sources
                first_chunk = False
            else:
                yield chunk, [], []

    def _build_context(
        self, relevant_docs: List[Tuple[Document, float]]
    ) -> str:
        """
        Build context string from retrieved documents.

        Args:
            relevant_docs: List of (document, similarity_score) tuples

        Returns:
            Formatted context string
        """
        if not relevant_docs:
            return ""

        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            context_parts.append(f"[Source {i}] (Relevance: {score:.2%})")
            if doc.file_name:
                context_parts.append(f"From: {doc.file_name}")
            context_parts.append(doc.content)
            context_parts.append("")  # Empty line between sources

        return "\n".join(context_parts)

    def add_document(self, document: Document) -> None:
        """
        Add a document to the RAG system.

        Args:
            document: The document to add
        """
        self.document_repository.save(document)

    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the system.

        Returns:
            List of all documents
        """
        return self.document_repository.get_all()

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the system.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if deleted successfully
        """
        return self.document_repository.delete(document_id)
