# RAG Application - Complete Project Guide

Welcome to the comprehensive RAG (Retrieval-Augmented Generation) application! This guide will help you navigate the entire project.

## Quick Start

### For First-Time Users

1. **Read**: `README.md` - Project overview
2. **Setup**: Follow installation instructions
3. **Run**: `streamlit run src/presentation/ui/app.py`
4. **Learn**: Start with `docs/guides/01_novice_guide.md`

### For Developers

1. **Architecture**: See `docs/architecture/architecture_overview.md`
2. **Code**: Explore `src/` with DDD structure
3. **Professional Guide**: Read `docs/guides/02_professional_guide.md`

### For Learners

1. **Novice Guide**: `docs/guides/01_novice_guide.md` - Start here!
2. **Transcripts**: `docs/transcripts/` - Conversational walkthroughs
3. **FAQ**: `docs/guides/04_faq.md` - Common questions
4. **Resources**: `docs/resources/learning_resources.md` - Further reading

## Project Structure

```
102-Claude-AskToCreateRAG/
│
├── README.md                          # Project overview and quick start
├── PROJECT_GUIDE.md                   # This file - navigation guide
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── src/                              # Source code (DDD architecture)
│   ├── domain/                       # Core business logic
│   │   ├── entities/                 # Document, Chat, Message
│   │   └── repositories/             # Repository interfaces
│   │
│   ├── application/                  # Use cases and orchestration
│   │   └── services/                 # RAG Service, Chat Service
│   │
│   ├── infrastructure/               # External integrations
│   │   ├── vector_store/             # ChromaDB implementation
│   │   ├── llm/                      # OpenAI service
│   │   └── document_loaders/         # PDF/text loaders
│   │
│   └── presentation/                 # User interface
│       └── ui/                       # Streamlit application
│
├── docs/                             # Comprehensive documentation
│   ├── architecture/                 # System design
│   │   ├── architecture_overview.md  # Architecture explanation
│   │   ├── c4_*.puml                 # C4 diagrams
│   │   ├── sequence_*.puml           # Sequence diagrams
│   │   └── ddd_entities.puml         # Entity diagrams
│   │
│   ├── guides/                       # Learning guides
│   │   ├── 01_novice_guide.md        # For beginners
│   │   ├── 02_professional_guide.md  # For developers
│   │   ├── 03_philosophical_foundation.md  # Design rationale
│   │   └── 04_faq.md                 # Common questions
│   │
│   ├── transcripts/                  # Conversational walkthroughs
│   │   ├── senior_to_junior_walkthrough.md  # Code walkthrough
│   │   └── engineer_to_philosopher.md       # Philosophical discussion
│   │
│   └── resources/                    # Learning materials
│       └── learning_resources.md     # Papers, tutorials, books
│
├── agentic_work/                     # Development documentation
│   ├── README.md                     # Agentic work overview
│   ├── thinking_process/             # Development reasoning
│   └── decision_log/                 # Key decisions made
│
└── data/                             # Runtime data (created on first run)
    └── chroma/                       # ChromaDB storage
```

## Documentation Map

### By Audience

**Novice Users** (No programming background):
1. `README.md` - Overview
2. `docs/guides/01_novice_guide.md` - Complete beginner guide
3. `docs/guides/04_faq.md` - Common questions
4. `docs/transcripts/engineer_to_philosopher.md` - Concepts explained

**Developers** (Want to use/extend):
1. `README.md` - Setup instructions
2. `docs/guides/02_professional_guide.md` - Technical deep dive
3. `docs/transcripts/senior_to_junior_walkthrough.md` - Code walkthrough
4. `docs/architecture/architecture_overview.md` - System design
5. `docs/guides/04_faq.md` - Technical FAQ

**Architects** (Want to understand design):
1. `docs/guides/03_philosophical_foundation.md` - Design philosophy
2. `docs/architecture/` - All architecture diagrams
3. `docs/guides/02_professional_guide.md` - Technical decisions
4. `agentic_work/README.md` - Development process

**Students** (Want to learn RAG):
1. `docs/guides/01_novice_guide.md` - Fundamentals
2. `docs/resources/learning_resources.md` - Papers and tutorials
3. `docs/transcripts/` - Different perspectives
4. Source code - Well-commented implementation

### By Topic

**RAG Fundamentals**:
- `docs/guides/01_novice_guide.md` - How RAG works
- `docs/guides/02_professional_guide.md` - RAG implementation details
- `docs/resources/learning_resources.md` - RAG research papers

**Architecture & Design**:
- `docs/architecture/architecture_overview.md` - Overall design
- `docs/guides/03_philosophical_foundation.md` - Why these decisions
- `docs/architecture/*.puml` - Visual diagrams
- `agentic_work/README.md` - Decision rationale

**Code Understanding**:
- `docs/transcripts/senior_to_junior_walkthrough.md` - Code tour
- `src/` - Source code with extensive comments
- `docs/guides/02_professional_guide.md` - Implementation patterns

**Practical Usage**:
- `README.md` - Quick start
- `docs/guides/04_faq.md` - Common issues and solutions
- `.env.example` - Configuration options

## Key Concepts Explained

### Domain-Driven Design (DDD)

The project uses DDD with four layers:

1. **Domain Layer** (`src/domain/`)
   - Core business entities: `Document`, `Message`, `ChatSession`
   - Repository interfaces (anti-corruption layer)
   - No dependencies on infrastructure

2. **Application Layer** (`src/application/`)
   - `RAGService`: Orchestrates retrieval and generation
   - `ChatService`: Manages conversations
   - Coordinates domain and infrastructure

3. **Infrastructure Layer** (`src/infrastructure/`)
   - `ChromaDocumentRepository`: Vector storage implementation
   - `OpenAIService`: LLM integration
   - `DocumentLoader`: File processing

4. **Presentation Layer** (`src/presentation/`)
   - Streamlit UI
   - User interaction
   - Depends on application layer

**Why DDD?** Clear separation of concerns, testability, maintainability. See: `docs/guides/03_philosophical_foundation.md`

### RAG Pattern

**Three Steps**:
1. **Retrieval**: Find relevant document chunks using semantic search
2. **Augmentation**: Inject chunks as context into LLM prompt
3. **Generation**: LLM generates answer based on context

**Implementation**: See `src/application/services/rag_service.py:query()`

### Vector Embeddings

Text → Numbers that capture meaning. Similar meanings → Similar numbers.

**Used for**: Semantic search (finding "vacation policy" also finds "time off guidelines")

**Learn more**: `docs/guides/01_novice_guide.md` Section "Key Concepts"

## Diagrams Guide

### C4 Diagrams (System Architecture)

Located in `docs/architecture/`:

1. **c4_system_context.puml**: System and external dependencies
2. **c4_container.puml**: Major components (UI, services, databases)
3. **c4_component.puml**: Internal component structure

**View**: Use PlantUML viewer or https://www.plantuml.com/plantuml/

### Sequence Diagrams (Workflows)

1. **sequence_document_upload.puml**: How documents are processed
2. **sequence_rag_query.puml**: How queries are answered
3. **ddd_entities.puml**: Domain model structure

## Customization Guide

### Common Customizations

**1. Change Chunk Size**
- **File**: `src/infrastructure/vector_store/chroma_document_repository.py`
- **Line**: Initialize `RecursiveCharacterTextSplitter`
- **Default**: 1000 chars with 200 overlap

**2. Adjust Retrieval**
- **File**: `src/application/services/rag_service.py`
- **Parameters**: `top_k_retrieval`, `similarity_threshold`
- **Defaults**: 5 chunks, 0.5 threshold

**3. Switch LLM Model**
- **File**: `src/infrastructure/llm/openai_service.py`
- **Parameter**: `model` in `__init__`
- **Default**: "gpt-4-turbo-preview"

**4. UI Customization**
- **File**: `src/presentation/ui/app.py`
- **Areas**: CSS in `st.markdown()`, layout in `st.columns()`

### Adding Features

**New Document Type** (e.g., Word docs):
1. Add loader in `src/infrastructure/document_loaders/document_loader.py`
2. Update UI file uploader in `src/presentation/ui/app.py`
3. Add to allowed extensions

**New Retrieval Strategy**:
1. Extend `IDocumentRepository` interface in `src/domain/repositories/`
2. Implement in `ChromaDocumentRepository`
3. Use in `RAGService`

**Full guide**: `docs/guides/04_faq.md` - "How do I add new features?"

## Testing

Currently no test suite included, but architecture supports testing.

**Example test structure**:
```python
def test_rag_query():
    mock_repo = Mock(spec=IDocumentRepository)
    mock_llm = Mock(spec=OpenAIService)

    rag_service = RAGService(mock_repo, mock_llm)
    response, sources = rag_service.query("test")

    assert response is not None
```

**See**: `docs/guides/02_professional_guide.md` - "Testing Strategy"

## Deployment

**Development**: `streamlit run src/presentation/ui/app.py`

**Docker**:
```bash
docker build -t rag-app .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... rag-app
```

**Production considerations**: See `docs/guides/02_professional_guide.md` - "Deployment" section

## Getting Help

### Documentation Hierarchy

1. **Quick answer**: `docs/guides/04_faq.md`
2. **Concept explanation**: `docs/guides/01_novice_guide.md`
3. **Technical detail**: `docs/guides/02_professional_guide.md`
4. **Design rationale**: `docs/guides/03_philosophical_foundation.md`
5. **Code walkthrough**: `docs/transcripts/senior_to_junior_walkthrough.md`

### Common Questions

**Q: How do I start?**
A: Read `README.md`, follow setup, run the app, then read novice guide.

**Q: How does RAG work?**
A: `docs/guides/01_novice_guide.md` - "How Does RAG Work?" section

**Q: Why this architecture?**
A: `docs/guides/03_philosophical_foundation.md` - Complete rationale

**Q: How do I modify X?**
A: `docs/guides/04_faq.md` - Check FAQ first, then professional guide

**Q: Where do I learn more about RAG?**
A: `docs/resources/learning_resources.md` - Curated papers and tutorials

## Learning Paths

### Path 1: Just Use It (1 hour)
1. Setup and run (30 min)
2. Read README and FAQ (30 min)
3. Upload documents and ask questions

### Path 2: Understand It (1 week)
1. Day 1: Novice guide
2. Day 2: Run and experiment
3. Day 3: Senior-to-junior transcript
4. Day 4: Professional guide (skim)
5. Day 5: Explore source code
6. Day 6: Modify something small
7. Day 7: Review architecture docs

### Path 3: Master It (1 month)
1. Week 1: All documentation
2. Week 2: Understand every line of code
3. Week 3: Read research papers (learning resources)
4. Week 4: Build a feature or variant

### Path 4: Teach It (Ongoing)
1. Master the content
2. Read philosophical foundation
3. Study agentic work documentation
4. Create your own RAG variant
5. Contribute improvements

## Philosophy

This project is built on principles:

**Educational First**: Code is clear, not clever
**Well-Documented**: Multiple perspectives for different learners
**Production-Ready**: Clean architecture, best practices
**Extensible**: Easy to modify and enhance

**Full philosophy**: `docs/guides/03_philosophical_foundation.md`

## Contributing

Improvements welcome!

**Good contributions**:
- Additional document loaders
- Test suite
- Performance optimizations
- Documentation clarifications
- Bug fixes

**Before contributing**:
1. Read professional guide
2. Understand DDD architecture
3. Follow existing patterns
4. Add tests
5. Update documentation

## Credits

Built with:
- **Python**: Language
- **LangChain**: RAG framework
- **ChromaDB**: Vector database
- **OpenAI**: LLM provider
- **Streamlit**: UI framework

Inspired by:
- RAG research papers
- Domain-Driven Design principles
- Clean Architecture patterns
- Open-source community

## License

Provided for educational and commercial use.

## Final Thoughts

This project demonstrates:
- **Technical**: Production-ready RAG implementation
- **Educational**: Comprehensive learning resource
- **Architectural**: Clean, maintainable design
- **Philosophical**: Thoughtful, principled decisions

Whether you're learning RAG, building applications, or studying software architecture, this codebase has something for you.

**Start exploring and happy learning!**
