# RAG Application - Exemplary Learning Project

A comprehensive Retrieval-Augmented Generation (RAG) application built with Python, LangChain, ChromaDB, and OpenAI. Designed as both a learning resource and a production-ready foundation for RAG applications.

## 🎯 Vision

Fill the jagged intelligence gap provided by LLMs for technical utility through an exemplary, feature-rich RAG implementation.

## ✨ Features

- **Multi-format Document Support**: PDF, text files, and direct text input
- **Intelligent Retrieval**: ChromaDB-powered semantic search with relevance scoring
- **Graph RAG**: Neo4j-powered graph traversal for enriched context retrieval
- **Hybrid Retrieval**: Combines vector search with graph relationships
- **Dynamic Entity Extraction**: Automatically builds ontology from document content
- **Conversational Interface**: WhatsApp-style chat with history
- **Source Attribution**: Track which documents informed each response (vector vs graph sources)
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

- **Python 3.12 or 3.13** (required - Python 3.14 is not yet supported due to onnxruntime dependency)
- OpenAI API key
- Docker (optional, for Graph RAG with Neo4j)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG
```

2. Create virtual environment with Python 3.12 or 3.13:
```bash
# Using Python 3.12 (recommended)
python3.12 -m venv venv
# OR using Python 3.13
python3.13 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you're using Python 3.14, you'll need to use Python 3.12 or 3.13 instead, as `onnxruntime` (required by `chromadb`) doesn't have wheels for Python 3.14 yet.

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Application

**Option 1: Using the startup script (Recommended)**
```bash
./start_app.sh
```

**Option 2: Manual start**
```bash
# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH and run
export PYTHONPATH="${PWD}:${PYTHONPATH}"
streamlit run src/presentation/ui/app.py
```

The application will open in your browser at `http://localhost:8501`

**Note**: Make sure you have a `.env` file with your `OPENAI_API_KEY` set, or export it as an environment variable.

### Enabling Graph RAG (Optional)

For enhanced retrieval with graph-based context enrichment:

1. **Start Neo4j**:
```bash
./start_neo4j.sh
# OR
docker-compose up -d
```

2. **Access Neo4j Browser** (optional):
   - Visit http://localhost:7474
   - Default credentials: `neo4j` / `password`

The application will automatically detect Neo4j and enable Graph RAG. If Neo4j is unavailable, it will fall back to vector-only retrieval.

See [GRAPH_RAG_SETUP.md](GRAPH_RAG_SETUP.md) for detailed Graph RAG documentation.

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

This application implements both classic and Graph RAG patterns:

### Classic RAG (Vector-Only)

1. **Document Ingestion**:
   - Documents are split into chunks (1000 chars with 200 char overlap)
   - Chunks are embedded and stored in ChromaDB

2. **Query Processing**:
   - User query is embedded
   - Top-K similar chunks are retrieved (K=5)
   - Chunks with similarity > 0.4 are used

3. **Response Generation**:
   - Retrieved context is injected into the system prompt
   - Chat history (last 5 messages) provides conversation context
   - OpenAI generates a contextually-aware response

### Graph RAG (Hybrid - When Neo4j is Available)

1. **Document Ingestion**:
   - Same as classic RAG (chunks stored in ChromaDB)
   - **Additionally**: Entities are extracted from each chunk using LLM
   - **Additionally**: Relationships between entities are identified
   - **Additionally**: Graph structure is built in Neo4j (Document → Chunk → Entity)

2. **Query Processing**:
   - **Step 1**: Vector search finds similar chunks (same as classic RAG)
   - **Step 2**: Entities are extracted from query and retrieved chunks
   - **Step 3**: Graph traversal finds related chunks through:
     - Chunks mentioning the same entities
     - Chunks connected via entity relationships
     - Chunks that co-occur with retrieved chunks
   - **Step 4**: Vector and graph results are combined and deduplicated

3. **Response Generation**:
   - Enriched context (vector + graph) is sent to LLM
   - Sources are labeled as "vector" or "graph" for transparency

## 📚 Documentation

Comprehensive documentation is available:

- **[Graph RAG Setup Guide](GRAPH_RAG_SETUP.md)**: Complete guide to Graph RAG setup and usage
- **Architecture Docs** (`docs/` directory):
  - **Novice Guide**: Concepts explained from basics
  - **Professional Guide**: Technical implementation details
  - **Philosophical Foundation**: Design decisions and their rationale
  - **Code Walkthrough**: Line-by-line explanation
  - **FAQ**: Common questions and answers
  - **Transcripts**: Conversational explorations of the code

## 🛠️ Technology Stack

- **Language**: Python 3.12+
- **LLM Framework**: LangChain
- **Vector Database**: ChromaDB
- **Graph Database**: Neo4j (optional, for Graph RAG)
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
