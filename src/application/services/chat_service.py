"""
Chat Service - Application layer for managing chat sessions.

This service manages chat sessions and integrates with the RAG service
to provide conversational interactions with document context.
"""
from typing import Optional
from src.domain.entities.chat import ChatSession, Message
from src.application.services.rag_service import RAGService


class ChatService:
    """
    Service for managing chat sessions and conversations.

    This service coordinates between chat sessions and the RAG service
    to provide a conversational interface to the document knowledge base.
    """

    def __init__(self, rag_service: RAGService):
        """
        Initialize the chat service.

        Args:
            rag_service: The RAG service for query processing
        """
        self.rag_service = rag_service
        self.current_session: Optional[ChatSession] = None

    def create_session(self) -> ChatSession:
        """
        Create a new chat session.

        Returns:
            The newly created chat session
        """
        self.current_session = ChatSession()
        return self.current_session

    def get_current_session(self) -> Optional[ChatSession]:
        """
        Get the current chat session.

        Returns:
            The current session, or None if no session exists
        """
        return self.current_session

    def send_message(self, content: str) -> Message:
        """
        Send a message and get a response.

        This method:
        1. Creates a user message
        2. Adds it to the session
        3. Queries the RAG service
        4. Creates an assistant message with the response
        5. Adds it to the session
        6. Returns the assistant message

        Args:
            content: The user's message content

        Returns:
            The assistant's response message
        """
        if not self.current_session:
            self.create_session()

        # Create and add user message
        user_message = Message(role="user", content=content)
        self.current_session.add_message(user_message)

        # Get chat history (excluding the just-added message for context)
        history = self.current_session.get_history()[:-1]

        # Query RAG service
        response_text, source_ids = self.rag_service.query(
            question=content, chat_history=history
        )

        # Create and add assistant message
        assistant_message = Message(
            role="assistant", content=response_text, sources=source_ids
        )
        self.current_session.add_message(assistant_message)

        return assistant_message

    def send_message_stream(self, content: str):
        """
        Send a message and stream the response.

        This is similar to send_message() but yields response chunks
        as they're generated.

        Args:
            content: The user's message content

        Yields:
            Response chunks and the final assistant message
        """
        if not self.current_session:
            self.create_session()

        # Create and add user message
        user_message = Message(role="user", content=content)
        self.current_session.add_message(user_message)

        # Get chat history
        history = self.current_session.get_history()[:-1]

        # Stream response from RAG service
        full_response = ""
        source_ids = []

        for chunk, sources in self.rag_service.query_stream(
            question=content, chat_history=history
        ):
            full_response += chunk
            if sources:  # Sources come with first chunk
                source_ids = sources
            yield chunk

        # Create and add assistant message
        assistant_message = Message(
            role="assistant", content=full_response, sources=source_ids
        )
        self.current_session.add_message(assistant_message)

        # Yield the complete message at the end
        yield assistant_message

    def clear_session(self) -> None:
        """
        Clear the current chat session.
        """
        self.current_session = None

    def get_message_count(self) -> int:
        """
        Get the number of messages in the current session.

        Returns:
            Number of messages, or 0 if no session
        """
        if not self.current_session:
            return 0
        return len(self.current_session.messages)
