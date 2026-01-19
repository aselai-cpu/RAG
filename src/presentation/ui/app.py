"""
Streamlit RAG Application - Main UI Entry Point

This is the presentation layer that provides a user-friendly interface
for interacting with the RAG system. Features two vertical panels:
1. Left: Document upload and management
2. Right: Chat interface (WhatsApp-style)
"""
import streamlit as st
import os
from pathlib import Path

# Import our application services
from src.application.services.rag_service import RAGService
from src.application.services.chat_service import ChatService
from src.application.services.hybrid_retrieval_service import HybridRetrievalService
from src.application.services.entity_extraction_service import EntityExtractionService
from src.infrastructure.vector_store.chroma_document_repository import (
    ChromaDocumentRepository,
)
from src.infrastructure.graph_store.neo4j_repository import Neo4jRepository
from src.infrastructure.llm.openai_service import OpenAIService
from src.infrastructure.document_loaders.document_loader import (
    DocumentLoader,
    DocumentValidator,
)
from src.infrastructure.logging import RAGLogger

# Get loggers for different aspects
user_logger = RAGLogger.get_logger('user_actions')
app_logger = RAGLogger.get_logger('rag_app')

# Page configuration
st.set_page_config(
    page_title="RAG Application",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .document-item {
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize services (singleton pattern using st.session_state)
@st.cache_resource
def initialize_services():
    """Initialize all application services."""
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "⚠️ OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )
        st.stop()

    # Initialize infrastructure layer
    document_repo = ChromaDocumentRepository(
        collection_name="rag_documents",
        persist_directory="./data/chroma",
        chunk_size=6000,
        chunk_overlap=600,
    )
    llm_service = OpenAIService(api_key=api_key)

    # Initialize Graph RAG services (optional - will fall back to vector-only if Neo4j unavailable)
    hybrid_retrieval_service = None
    try:
        neo4j_repo = Neo4jRepository()
        entity_extraction_service = EntityExtractionService(llm_service=llm_service)
        hybrid_retrieval_service = HybridRetrievalService(
            document_repository=document_repo,
            neo4j_repository=neo4j_repo,
            entity_extraction_service=entity_extraction_service,
        )
        app_logger.info("Graph RAG enabled - Neo4j connection successful")
    except Exception as e:
        app_logger.warning(f"Graph RAG disabled - Neo4j unavailable: {str(e)}")
        st.sidebar.warning(
            "⚠️ Graph RAG disabled. Neo4j not available. "
            "Using vector-only retrieval. Start Neo4j with: docker-compose up -d"
        )

    # Initialize application layer
    rag_service = RAGService(
        document_repository=document_repo,
        llm_service=llm_service,
        hybrid_retrieval_service=hybrid_retrieval_service,
    )
    chat_service = ChatService(rag_service=rag_service)

    return rag_service, chat_service


# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chat_session_initialized" not in st.session_state:
        st.session_state.chat_session_initialized = False


# Document upload panel
def document_panel(rag_service):
    """Render the document upload and management panel."""
    st.header("📚 Documents")

    # Upload options
    upload_method = st.radio(
        "Upload Method",
        ["Upload File", "Paste Text"],
        horizontal=True,
    )

    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt"],
            help="Upload PDF or text files (max 10MB)",
        )

        if uploaded_file is not None:
            if st.button("📤 Upload Document"):
                user_logger.info(f"User clicked 'Upload Document' button: filename={uploaded_file.name}, size={uploaded_file.size} bytes")
                with st.spinner("Processing document..."):
                    try:
                        # Read file bytes
                        file_bytes = uploaded_file.read()

                        # Validate file
                        DocumentValidator.validate_file_size(file_bytes)
                        DocumentValidator.validate_file_type(
                            uploaded_file.name, [".pdf", ".txt"]
                        )
                        user_logger.info(f"File validation passed: filename={uploaded_file.name}")

                        # Load document based on type
                        if uploaded_file.name.endswith(".pdf"):
                            user_logger.info(f"Loading PDF document: {uploaded_file.name}")
                            document = DocumentLoader.load_from_pdf(
                                file_bytes, uploaded_file.name
                            )
                        else:
                            user_logger.info(f"Loading text document: {uploaded_file.name}")
                            document = DocumentLoader.load_from_text(
                                file_bytes, uploaded_file.name
                            )

                        # Add to RAG system
                        user_logger.info(f"Adding document to RAG system: doc_id={document.id}, filename={uploaded_file.name}")
                        rag_service.add_document(document)

                        # Update session state
                        st.session_state.documents.append(document)

                        user_logger.info(f"Document uploaded successfully: doc_id={document.id}, filename={uploaded_file.name}, chunks={document.chunk_count}")
                        st.success(f"✅ Uploaded: {uploaded_file.name}")
                        st.rerun()

                    except Exception as e:
                        user_logger.error(f"Error uploading document: filename={uploaded_file.name}, error={str(e)}", exc_info=True)
                        st.error(f"❌ Error: {str(e)}")

    else:  # Paste Text
        pasted_text = st.text_area(
            "Paste your text here",
            height=200,
            placeholder="Paste or type your text content here...",
        )

        if st.button("📤 Add Text"):
            if pasted_text.strip():
                user_logger.info(f"User clicked 'Add Text' button: text_length={len(pasted_text)} chars")
                with st.spinner("Processing text..."):
                    try:
                        # Load from clipboard
                        document = DocumentLoader.load_from_clipboard(pasted_text)

                        # Add to RAG system
                        user_logger.info(f"Adding pasted text to RAG system: doc_id={document.id}, length={len(pasted_text)}")
                        rag_service.add_document(document)

                        # Update session state
                        st.session_state.documents.append(document)

                        user_logger.info(f"Text added successfully: doc_id={document.id}, chunks={document.chunk_count}")
                        st.success("✅ Text added successfully!")
                        st.rerun()

                    except Exception as e:
                        user_logger.error(f"Error adding pasted text: error={str(e)}", exc_info=True)
                        st.error(f"❌ Error: {str(e)}")
            else:
                user_logger.warning("User clicked 'Add Text' but no text was pasted")
                st.warning("⚠️ Please paste some text first")

    # Display uploaded documents
    st.divider()
    st.subheader("Uploaded Documents")

    # Refresh documents list
    if st.button("🔄 Refresh"):
        st.session_state.documents = rag_service.get_all_documents()
        st.rerun()

    if st.session_state.documents:
        for doc in st.session_state.documents:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{doc.get_display_name()}**")
                    st.caption(
                        f"{doc.source_type.upper()} • {doc.chunk_count} chunks • {len(doc.content)} chars"
                    )
                with col2:
                    if st.button("🗑️", key=f"delete_{doc.id}"):
                        rag_service.delete_document(doc.id)
                        st.session_state.documents = [
                            d for d in st.session_state.documents if d.id != doc.id
                        ]
                        st.rerun()
                
                # View ChromaDB chunks for this document
                with st.expander(f"🔍 View ChromaDB Chunks ({doc.chunk_count})", expanded=False):
                    if hasattr(rag_service.document_repository, 'get_chunks_by_document'):
                        chunks = rag_service.document_repository.get_chunks_by_document(doc.id, limit=20)
                        if chunks:
                            for chunk in chunks:
                                st.markdown(f"**Chunk {chunk['chunk_index']}** (ID: `{chunk['chunk_id']}`)")
                                st.text_area(
                                    f"Content",
                                    value=chunk['content'],
                                    height=100,
                                    key=f"chunk_{chunk['chunk_id']}",
                                    label_visibility="collapsed"
                                )
                                # Show metadata in a collapsible format without nested expander
                                st.markdown("**Metadata:**")
                                st.json(chunk['metadata'])
                                st.divider()
                        else:
                            st.info("No chunks found in ChromaDB for this document.")
                
                st.divider()
    else:
        st.info("📝 No documents uploaded yet. Upload documents to get started!")
    
    # ChromaDB Collection Info
    st.divider()
    with st.expander("🗄️ ChromaDB Collection Info", expanded=False):
        if hasattr(rag_service.document_repository, 'get_collection_info'):
            collection_info = rag_service.document_repository.get_collection_info()
            st.metric("Total Chunks", collection_info.get("total_chunks", 0))
            st.metric("Total Documents", collection_info.get("total_documents", 0))
            st.caption(f"Collection: `{collection_info.get('collection_name', 'N/A')}`")
            if "error" in collection_info:
                st.error(f"Error: {collection_info['error']}")


# Chat panel
def chat_panel(chat_service):
    """Render the chat interface panel."""
    st.header("💬 Chat")

    # Initialize chat session if needed
    if not st.session_state.chat_session_initialized:
        chat_service.create_session()
        st.session_state.chat_session_initialized = True

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            sources = message.get("sources", [])

            with st.chat_message(role):
                st.markdown(content)
                if sources and role == "assistant":
                    with st.expander("📎 Sources"):
                        # Handle both old format (list of IDs) and new format (list of dicts)
                        for source in sources:
                            if isinstance(source, dict):
                                # New format with detailed information
                                doc_id = source.get("document_id", "Unknown")
                                chunk_index = source.get("chunk_index", "?")
                                relevance = source.get("relevance_score", 0)
                                file_name = source.get("file_name", "")
                                
                                # Format relevance as percentage
                                relevance_pct = f"{relevance * 100:.1f}%"
                                
                                # Display with file name if available
                                source_type = source.get("source_type", "vector")
                                source_icon = "🕸️" if source_type == "graph" else "📊"
                                if file_name:
                                    st.caption(f"{source_icon} **{file_name}** - Chunk {chunk_index} (Relevance: {relevance_pct}) [{source_type}]")
                                    st.caption(f"  Document ID: `{doc_id}`")
                                else:
                                    st.caption(f"{source_icon} Document ID: `{doc_id}` - Chunk {chunk_index} (Relevance: {relevance_pct}) [{source_type}]")
                            else:
                                # Old format - just document ID (backward compatibility)
                                st.caption(f"• Document ID: {source}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        user_logger.info(f"User submitted question: query_length={len(prompt)}, query='{prompt[:200]}...'")

        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Stream response
                for chunk in chat_service.send_message_stream(prompt):
                    # Check if it's the final message object
                    if hasattr(chunk, "content"):
                        # This is the final Message object
                        sources = chunk.sources
                        # Add to session state
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": full_response,
                                "sources": sources,
                            }
                        )
                        user_logger.info(f"Response generated: response_length={len(full_response)}, num_sources={len(sources)}")
                        # Display sources
                        if sources:
                            with st.expander("📎 Sources"):
                                # Handle both old format (list of IDs) and new format (list of dicts)
                                for source in sources:
                                    if isinstance(source, dict):
                                        # New format with detailed information
                                        doc_id = source.get("document_id", "Unknown")
                                        chunk_index = source.get("chunk_index", "?")
                                        relevance = source.get("relevance_score", 0)
                                        file_name = source.get("file_name", "")
                                        
                                        # Format relevance as percentage
                                        relevance_pct = f"{relevance * 100:.1f}%"
                                        
                                        # Display with file name if available
                                        source_type = source.get("source_type", "vector")
                                        source_icon = "🕸️" if source_type == "graph" else "📊"
                                        if file_name:
                                            st.caption(f"{source_icon} **{file_name}** - Chunk {chunk_index} (Relevance: {relevance_pct}) [{source_type}]")
                                            st.caption(f"  Document ID: `{doc_id}`")
                                        else:
                                            st.caption(f"{source_icon} Document ID: `{doc_id}` - Chunk {chunk_index} (Relevance: {relevance_pct}) [{source_type}]")
                                    else:
                                        # Old format - just document ID (backward compatibility)
                                        st.caption(f"• Document ID: {source}")
                    else:
                        # This is a text chunk
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            except Exception as e:
                user_logger.error(f"Error generating response: error={str(e)}", exc_info=True)
                st.error(f"❌ Error generating response: {str(e)}")

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        user_logger.info(f"User clicked 'Clear Chat' button: message_count={len(st.session_state.messages)}")
        st.session_state.messages = []
        chat_service.clear_session()
        st.session_state.chat_session_initialized = False
        st.rerun()


# Main application
def main():
    """Main application entry point."""
    # Initialize logging system first
    RAGLogger.setup_logging()
    app_logger.info("Starting RAG Application")

    # Initialize
    initialize_session_state()
    rag_service, chat_service = initialize_services()

    # Title
    st.title("🤖 RAG Application")
    st.caption("Retrieval-Augmented Generation with Document Understanding")

    # Two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        document_panel(rag_service)

    with col2:
        chat_panel(chat_service)

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
        This RAG application allows you to:
        - Upload PDF and text documents
        - Paste text directly
        - Ask questions about your documents
        - Get AI-powered answers with sources

        **Features:**
        - Vector-based semantic search (ChromaDB)
        - Graph RAG with Neo4j (entity extraction & graph traversal)
        - Hybrid retrieval combining vector + graph
        - OpenAI-powered responses
        - Source attribution
        - Chat history
        """
        )

        st.divider()

        st.header("📊 Statistics")
        st.metric("Documents", len(st.session_state.documents))
        st.metric("Chat Messages", len(st.session_state.messages))
        
        # ChromaDB Collection Statistics
        if hasattr(rag_service.document_repository, 'get_collection_info'):
            st.divider()
            st.header("🗄️ ChromaDB")
            collection_info = rag_service.document_repository.get_collection_info()
            st.metric("Total Chunks", collection_info.get("total_chunks", 0))
            st.metric("Total Documents", collection_info.get("total_documents", 0))
            st.caption(f"Collection: `{collection_info.get('collection_name', 'N/A')}`")
        
        # Neo4j Graph Statistics
        if rag_service.use_graph_rag and rag_service.hybrid_retrieval_service:
            st.divider()
            st.header("🕸️ Neo4j Graph")
            try:
                graph_stats = rag_service.hybrid_retrieval_service.neo4j_repository.get_graph_stats()
                st.metric("Documents", graph_stats.get("documents", 0))
                st.metric("Chunks", graph_stats.get("chunks", 0))
                st.metric("Entities", graph_stats.get("entities", 0))
                st.metric("Relationships", graph_stats.get("relationships", 0))
            except Exception as e:
                st.error(f"Error getting graph stats: {str(e)}")


if __name__ == "__main__":
    main()
