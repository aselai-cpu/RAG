# RAG Application - Exemplary Learning Project

A comprehensive Retrieval-Augmented Generation (RAG) application built with Python, LangChain, ChromaDB, and OpenAI. Designed as both a learning resource and a production-ready foundation for RAG applications.

## 🎯 Vision

Fill the jagged intelligence gap provided by LLMs for technical utility through an exemplary, feature-rich RAG implementation.

## ✨ Features

- **Multi-format Document Support**: PDF, text files, and direct text input
- **Intelligent Retrieval**: ChromaDB-powered semantic search with relevance scoring
- **Conversational Interface**: WhatsApp-style chat with history
- **Source Attribution**: Track which documents informed each response
- **Domain-Driven Design**: Clean architecture with separation of concerns
- **Streamlit UI**: Intuitive two-panel interface

## 🏗️ Architecture

This project follows Domain-Driven Design (DDD) principles:

```
src/
├── domain/              # Core business logic
│   ├── entities/        # Document, Chat, Message
│   └── repositories/    # Repository interfaces (Anti-Corruption Layer)
├── application/         # Use cases and services
│   └── services/        # RAG and Chat orchestration
├── infrastructure/      # External integrations
│   ├── vector_store/    # ChromaDB implementation
│   ├── llm/             # OpenAI service
│   └── document_loaders/# Document processing
└── presentation/        # UI layer
    └── ui/              # Streamlit application
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 102-Claude-AskToCreateRAG
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Application

```bash
streamlit run src/presentation/ui/app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Upload Documents**:
   - Click "Upload File" to add PDF or text files
   - Or use "Paste Text" to directly input content

2. **Ask Questions**:
   - Type your question in the chat input
   - The system will retrieve relevant context and generate an informed response
   - View sources to see which documents were used

3. **Manage Documents**:
   - View all uploaded documents in the left panel
   - Delete documents as needed
   - Refresh to sync the document list

## 🧪 RAG Workflow

This application implements the classic RAG pattern:

1. **Document Ingestion**:
   - Documents are split into chunks (1000 chars with 200 char overlap)
   - Chunks are embedded and stored in ChromaDB

2. **Query Processing**:
   - User query is embedded
   - Top-K similar chunks are retrieved (K=5)
   - Chunks with similarity > 0.5 are used

3. **Response Generation**:
   - Retrieved context is injected into the system prompt
   - Chat history (last 5 messages) provides conversation context
   - OpenAI generates a contextually-aware response

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Novice Guide**: Concepts explained from basics
- **Professional Guide**: Technical implementation details
- **Philosophical Foundation**: Design decisions and their rationale
- **Code Walkthrough**: Line-by-line explanation
- **FAQ**: Common questions and answers
- **Transcripts**: Conversational explorations of the code

## 🛠️ Technology Stack

- **Language**: Python 3.9+
- **LLM Framework**: LangChain
- **Vector Database**: ChromaDB
- **LLM Provider**: OpenAI (GPT-4)
- **UI Framework**: Streamlit
- **Document Processing**: PyPDF2

## 🔧 Configuration

Key configuration options (in service initialization):

- `chunk_size`: Text chunk size (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `top_k_retrieval`: Number of chunks to retrieve (default: 5)
- `similarity_threshold`: Minimum similarity for relevance (default: 0.5)
- `model`: OpenAI model (default: gpt-4-turbo-preview)
- `temperature`: Response randomness (default: 0.7)

## 🏆 Best Practices (2025/26)

This implementation incorporates current RAG best practices:

1. **Chunking Strategy**: Recursive character splitting with overlap
2. **Embedding Model**: OpenAI's latest embeddings
3. **Retrieval Method**: Semantic similarity with threshold filtering
4. **Context Management**: Limited context window with recent history
5. **Source Attribution**: Track and display source documents
6. **Clean Architecture**: DDD for maintainability and testability

## 🤝 Contributing

Contributions are welcome! This project serves as a learning resource, so improvements to documentation and code clarity are especially valued.

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

Built with modern RAG principles and best practices from the ML community.
