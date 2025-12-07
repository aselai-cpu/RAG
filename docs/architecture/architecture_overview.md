# RAG Application - Architecture Overview

## Introduction

This document provides a comprehensive overview of the RAG (Retrieval-Augmented Generation) application architecture. The system is built following Domain-Driven Design (DDD) principles with a clean, layered architecture.

## Architectural Principles

### 1. Domain-Driven Design (DDD)

The application is structured around core domain concepts:

- **Entities**: `Document`, `Message`, `ChatSession` - representing core business objects
- **Repository Interfaces**: Abstract contracts for data access (Anti-Corruption Layer)
- **Services**: Orchestrate domain logic and use cases

### 2. Layered Architecture

```
┌─────────────────────────────────────┐
│     Presentation Layer              │  <- Streamlit UI
├─────────────────────────────────────┤
│     Application Layer               │  <- Services & Use Cases
├─────────────────────────────────────┤
│     Domain Layer                    │  <- Entities & Interfaces
├─────────────────────────────────────┤
│     Infrastructure Layer            │  <- ChromaDB, OpenAI, Loaders
└─────────────────────────────────────┘
```

Each layer has specific responsibilities:

- **Presentation**: User interface and interaction
- **Application**: Business workflows and orchestration
- **Domain**: Core business logic and rules
- **Infrastructure**: External systems and technical concerns

### 3. Dependency Inversion

The domain layer defines interfaces (`IDocumentRepository`) that are implemented by the infrastructure layer (`ChromaDocumentRepository`). This means:

- Domain doesn't depend on infrastructure
- Infrastructure depends on domain interfaces
- Easy to swap implementations (e.g., switch from ChromaDB to Pinecone)

## Core Components

### Domain Layer

**Entities:**
- `Document`: Represents a document with content, metadata, and source information
- `Message`: Represents a single chat message
- `ChatSession`: Aggregates messages into a conversation

**Repository Interfaces (Anti-Corruption Layer):**
- `IDocumentRepository`: Contract for document storage and retrieval
- `IChatRepository`: Contract for chat persistence

### Application Layer

**Services:**
- `RAGService`: Orchestrates the RAG workflow
  - Retrieves relevant documents
  - Builds context
  - Generates responses
- `ChatService`: Manages chat sessions
  - Creates and maintains sessions
  - Coordinates with RAG service
  - Handles message history

### Infrastructure Layer

**Implementations:**
- `ChromaDocumentRepository`: ChromaDB implementation of document repository
  - Handles text chunking
  - Manages embeddings
  - Performs semantic search
- `OpenAIService`: Wrapper for OpenAI API
  - Chat completions
  - Streaming responses
  - Token management
- `DocumentLoader`: Loads documents from various sources
  - PDF extraction
  - Text file reading
  - Clipboard input

### Presentation Layer

**Streamlit UI:**
- Two-panel layout
- Document upload and management
- Chat interface
- Real-time streaming responses

## RAG Workflow

The application implements the classic RAG pattern:

### 1. Document Ingestion

```
User Upload → DocumentLoader → Document Entity → RAG Service
                                                      ↓
                                         ChromaDocumentRepository
                                                      ↓
                                              Text Splitter
                                                      ↓
                                                  ChromaDB
                                            (Store embeddings)
```

**Steps:**
1. User uploads PDF/text or pastes content
2. DocumentLoader extracts text and creates Document entity
3. RAG Service passes document to repository
4. Repository splits text into chunks (1000 chars, 200 overlap)
5. Chunks are embedded and stored in ChromaDB

### 2. Query Processing

```
User Question → Chat Service → RAG Service
                                    ↓
                         Retrieval Phase
                                    ↓
                    ChromaDocumentRepository
                                    ↓
                              ChromaDB Query
                    (Semantic similarity search)
                                    ↓
                         Generation Phase
                                    ↓
                            OpenAI Service
                                    ↓
                         Augmented Prompt
                    (Context + Chat History)
                                    ↓
                            OpenAI API
                                    ↓
                         Stream Response
```

**Retrieval Phase:**
1. Query is embedded
2. Top-K similar chunks retrieved (K=5)
3. Filter by similarity threshold (>0.5)
4. Build context from relevant chunks

**Generation Phase:**
1. Inject context into system message
2. Add recent chat history (last 5 messages)
3. Send to OpenAI for completion
4. Stream response back to user
5. Track source document IDs

## Design Patterns

### 1. Repository Pattern

Abstracts data access behind interfaces:
- Domain defines `IDocumentRepository`
- Infrastructure implements `ChromaDocumentRepository`
- Application layer depends only on interface

**Benefits:**
- Easy to test (mock repositories)
- Easy to swap implementations
- Domain remains pure

### 2. Service Pattern

Application services orchestrate workflows:
- `RAGService` coordinates retrieval and generation
- `ChatService` manages conversation flow

### 3. Anti-Corruption Layer

Repository interfaces protect domain from external changes:
- ChromaDB API changes don't affect domain
- Domain speaks its own language
- Infrastructure translates

### 4. Dependency Injection

Services receive dependencies via constructor:
```python
RAGService(
    document_repository=ChromaDocumentRepository(),
    llm_service=OpenAIService()
)
```

**Benefits:**
- Testable (inject mocks)
- Flexible (swap implementations)
- Clear dependencies

## Technology Choices

### Why ChromaDB?

- **Lightweight**: Easy to set up and run locally
- **No external dependencies**: Perfect for learning
- **Python-native**: Seamless integration
- **Production-ready**: Can scale when needed

### Why OpenAI?

- **State-of-the-art**: Best LLM performance
- **Reliable embeddings**: High-quality semantic search
- **Streaming support**: Real-time user experience

### Why Streamlit?

- **Python-native**: No JavaScript needed
- **Rapid development**: Quick prototyping
- **Interactive**: Built-in components for chat
- **Learning-friendly**: Easy to understand

### Why LangChain?

- **Text splitting**: Robust chunking strategies
- **Abstractions**: Common RAG patterns
- **Community**: Best practices and examples

## Data Flow

### Document Storage

```
PDF/Text File
     ↓
[Extract Text]
     ↓
Document Entity (full text)
     ↓
[Split into chunks]
     ↓
Chunk 1, Chunk 2, ..., Chunk N
     ↓
[Generate embeddings]
     ↓
ChromaDB (vector storage)
```

### Query Processing

```
User Question
     ↓
[Embed query]
     ↓
Vector: [0.23, -0.15, ...]
     ↓
[Semantic search in ChromaDB]
     ↓
Top-K chunks with scores
     ↓
[Filter by threshold]
     ↓
Relevant chunks
     ↓
[Build context]
     ↓
Augmented prompt
     ↓
OpenAI API
     ↓
Streaming response
```

## Scalability Considerations

### Current Implementation (Learning & Prototyping)

- Local ChromaDB instance
- In-memory document cache
- Session state in Streamlit

### Production Enhancements

1. **Database**: Replace in-memory cache with persistent DB
2. **Vector Store**: Use managed ChromaDB or Pinecone
3. **Session Management**: Redis or database-backed sessions
4. **Async Processing**: Background document ingestion
5. **Caching**: Cache embeddings and common queries
6. **Monitoring**: Add observability and logging

## Security Considerations

1. **API Keys**: Stored in environment variables
2. **File Validation**: Size and type checking
3. **Input Sanitization**: Prevent injection attacks
4. **Rate Limiting**: Add for production use

## Testing Strategy

### Unit Tests
- Domain entities
- Service logic
- Document loading

### Integration Tests
- Repository implementations
- OpenAI service
- End-to-end RAG workflow

### UI Tests
- Streamlit components
- User interactions

## Future Enhancements

1. **Multiple LLM Support**: Anthropic, Cohere, local models
2. **Advanced Retrieval**: Hybrid search, re-ranking
3. **Multi-modal**: Image and audio documents
4. **Collaborative**: Multi-user sessions
5. **Analytics**: Usage tracking and insights

## Conclusion

This architecture balances:
- **Learning**: Clear, understandable structure
- **Best practices**: DDD, clean architecture
- **Production-ready**: Scalable foundations
- **Flexibility**: Easy to extend and modify

The Domain-Driven Design approach ensures the core business logic remains pure and testable, while the layered architecture provides clear separation of concerns.
