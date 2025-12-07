# Code Walkthrough: Senior Engineer to Junior Developer

**Participants**:
- **Sarah** (Senior Software Engineer, 10 years experience)
- **Alex** (Junior Developer, 6 months experience, knows Python basics)

**Setting**: Code review session for the RAG application

---

## Session Start

**Sarah**: Hey Alex! Thanks for joining me. I wanted to walk you through this RAG application we've built. Have you looked at the code yet?

**Alex**: I've skimmed through it, but honestly, I'm a bit overwhelmed. There are so many folders and files. Where do I even start?

**Sarah**: That's totally normal! Let's start with the big picture. See this folder structure?

```
src/
├── domain/
├── application/
├── infrastructure/
└── presentation/
```

Think of it like a layered cake. Each layer has a specific job.

**Alex**: OK, so like... different responsibilities?

**Sarah**: Exactly! Let me use an analogy. Imagine you're building a restaurant:

- **domain/**: The recipes and cooking rules (core business logic)
- **application/**: The head chef coordinating everything (orchestration)
- **infrastructure/**: The actual stoves, fridges, suppliers (external systems)
- **presentation/**: The dining room and waiters (user interface)

**Alex**: Oh! That makes sense. So we start with... domain?

## Understanding the Domain Layer

**Sarah**: Yep! Open `src/domain/entities/document.py`. What do you see?

**Alex**: A Python dataclass called `Document` with fields like `id`, `content`, `source_type`...

**Sarah**: Right. This is our core concept - what IS a document in our system? Notice the `__post_init__` method?

```python
def __post_init__(self):
    if not self.content:
        raise ValueError("Document content cannot be empty")
    if self.source_type not in ["pdf", "text", "clipboard"]:
        raise ValueError(f"Invalid source_type: {self.source_type}")
```

**Alex**: It's doing validation?

**Sarah**: Exactly! The Document entity knows its own rules. A document without content doesn't make sense, so it refuses to exist in that state. This is called "maintaining invariants."

**Alex**: So instead of checking "is this document valid?" everywhere in the code, the document checks itself?

**Sarah**: Bingo! You're getting it. Now look at `chat.py`. We have two entities there.

**Alex**: `Message` and `ChatSession`. Message is a single message, and ChatSession is... a collection of messages?

**Sarah**: Yup! And notice the `add_message` method:

```python
def add_message(self, message: Message) -> None:
    self.messages.append(message)
    self.updated_at = datetime.now()
```

**Alex**: It updates the timestamp automatically!

**Sarah**: Right. The ChatSession knows that when you add a message, you've updated the session. This knowledge lives in the entity, not scattered across the codebase.

## Repository Pattern

**Alex**: OK, so entities are our data structures. What about `domain/repositories/`?

**Sarah**: Great question! Open `document_repository.py`. What do you notice?

**Alex**: It's a class called `IDocumentRepository` with `ABC` and all the methods just say `pass`. Is this... incomplete?

**Sarah**: No, it's an interface! Notice the `ABC` - Abstract Base Class? This defines WHAT we need to do with documents, not HOW.

```python
class IDocumentRepository(ABC):
    @abstractmethod
    def save(self, document: Document) -> None:
        pass

    @abstractmethod
    def search_similar(self, query: str, top_k: int) -> List[tuple[Document, float]]:
        pass
```

**Alex**: So... anyone implementing this must provide these methods?

**Sarah**: Exactly! Now go to `infrastructure/vector_store/chroma_document_repository.py`.

**Alex**: Oh! `class ChromaDocumentRepository(IDocumentRepository):` - it implements the interface!

**Sarah**: You got it! This is the actual implementation using ChromaDB. But here's the cool part - the domain layer only knows about the interface. It doesn't care if we use ChromaDB, Pinecone, or even just files on disk.

**Alex**: So we could swap databases without changing the domain code?

**Sarah**: Exactly! That's the power of the Repository Pattern. It's an "anti-corruption layer" - it protects the domain from infrastructure changes.

**Alex**: Anti-corruption layer... that's a cool name. So if ChromaDB releases a breaking change, we only fix it in one place?

**Sarah**: Precisely! Now look at how `save` is implemented:

```python
def save(self, document: Document) -> None:
    # Split document into chunks
    chunks = self.text_splitter.split_text(document.content)
    document.chunk_count = len(chunks)

    # Prepare data for ChromaDB
    chunk_ids = [f"{document.id}_{i}" for i in range(len(chunks))]
    metadatas = [...]

    # Add to ChromaDB
    self.collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids,
    )
```

**Alex**: Wait, it's splitting the document into chunks? Why?

**Sarah**: Great question! Imagine you have a 100-page employee handbook. If someone asks "What's the vacation policy?", you don't want to search the entire handbook, right? You want to find the specific section about vacations.

**Alex**: Oh! So chunks are like... sections?

**Sarah**: Exactly! We split documents into ~1000 character chunks. This way, we can find precisely relevant information.

**Alex**: And the `chunk_ids` - that's like `document_id_0`, `document_id_1`?

**Sarah**: Yep! So we can trace chunks back to their source document.

## Application Layer - Where Things Happen

**Sarah**: OK, let's move up a layer. Open `application/services/rag_service.py`.

**Alex**: Whoa, this is longer...

**Sarah**: Don't worry, let's break it down. Find the `query` method. This is the heart of RAG.

```python
def query(self, question: str, chat_history=None) -> Tuple[str, List[str]]:
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

    # Step 2: Build context
    context = self._build_context(relevant_docs)

    # Step 3: Generate response
    response = self.llm_service.generate_response(
        messages=messages, context=context
    )

    return response, source_ids
```

**Alex**: So it's like three steps - retrieve, build context, generate?

**Sarah**: Exactly! This is the classic RAG pattern:
1. **R**etrieval - Find relevant chunks
2. **A**ugmented - Add them as context
3. **G**eneration - LLM generates answer

**Alex**: And the threshold filtering?

**Sarah**: We only keep chunks with similarity > 0.5 (50%). This filters out irrelevant stuff.

**Alex**: How does similarity work?

**Sarah**: It's called "cosine similarity." Think of it like measuring how similar two texts are in meaning. 1.0 means identical, 0.0 means completely different. 0.5 is moderately similar.

**Alex**: So we're not doing keyword matching? It's understanding meaning?

**Sarah**: Exactly! That's the magic of embeddings. "Remote work" and "work from home" have different words but similar embeddings.

**Alex**: That's so cool! What about `_build_context`?

**Sarah**: Look at the implementation:

```python
def _build_context(self, relevant_docs: List[Tuple[Document, float]]) -> str:
    context_parts = []
    for i, (doc, score) in enumerate(relevant_docs, 1):
        context_parts.append(f"[Source {i}] (Relevance: {score:.2%})")
        if doc.file_name:
            context_parts.append(f"From: {doc.file_name}")
        context_parts.append(doc.content)
        context_parts.append("")  # Empty line

    return "\n".join(context_parts)
```

**Alex**: It's building a string with all the relevant chunks, numbered and labeled?

**Sarah**: Yep! This gets sent to the LLM as context. The LLM reads this, then answers the user's question based on it.

**Alex**: So the LLM sees something like:

```
[Source 1] (Relevance: 85%)
From: employee_handbook.pdf
Vacation policy: Employees get 15 days PTO...

[Source 2] (Relevance: 72%)
From: hr_guidelines.txt
New employees accrue PTO starting from day one...
```

**Sarah**: Exactly! Then the LLM can say "According to the employee handbook, you get 15 days PTO."

## Infrastructure Layer - Talking to External Systems

**Alex**: Can we look at the OpenAI service?

**Sarah**: Sure! Open `infrastructure/llm/openai_service.py`. Check out `generate_response`:

```python
def generate_response(
    self,
    messages: List[Dict[str, str]],
    context: Optional[str] = None,
) -> str:
    # Build messages
    api_messages = []

    if context:
        system_message = {
            "role": "system",
            "content": f"""You are a helpful AI assistant...
Context:
{context}
""",
        }
        api_messages.append(system_message)

    api_messages.extend(messages)

    # Generate response
    response = self.client.chat.completions.create(
        model=self.model,
        messages=api_messages,
        temperature=self.temperature,
    )

    return response.choices[0].message.content
```

**Alex**: It's building a messages list and sending to OpenAI?

**Sarah**: Right! The system message includes our retrieved context. Then we add the conversation messages. The OpenAI API processes this and returns a response.

**Alex**: What's `temperature`?

**Sarah**: It controls randomness. 0.0 is very deterministic (same input = same output). 1.0 is very creative/random. We use 0.7 for a balance.

**Alex**: For factual Q&A, wouldn't we want lower temperature?

**Sarah**: Great thinking! You could totally set it to 0.2-0.3 for more consistent, factual responses. This is a tunable parameter.

**Alex**: And `generate_response_stream` does the same thing but yields chunks?

**Sarah**: Exactly! Instead of waiting for the full response, we stream it word-by-word. This makes the UI feel more responsive.

```python
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        yield chunk.choices[0].delta.content
```

**Alex**: So the UI can display it as it's being generated, like ChatGPT?

**Sarah**: Precisely!

## Presentation Layer - The User Interface

**Sarah**: Let's look at the UI. Open `presentation/ui/app.py`.

**Alex**: This is huge!

**Sarah**: Don't worry, it's mostly UI code. See the `initialize_services` function?

```python
@st.cache_resource
def initialize_services():
    # Initialize infrastructure
    document_repo = ChromaDocumentRepository(...)
    llm_service = OpenAIService(api_key=api_key)

    # Initialize application
    rag_service = RAGService(
        document_repository=document_repo,
        llm_service=llm_service,
    )
    chat_service = ChatService(rag_service=rag_service)

    return rag_service, chat_service
```

**Alex**: It's creating all the services... and passing the repository to RAGService!

**Sarah**: Right! This is Dependency Injection. We're saying "RAGService, here's your repository, here's your LLM service." RAGService doesn't create them itself.

**Alex**: Why?

**Sarah**: Two reasons:
1. **Testing**: We can inject fake/mock services for tests
2. **Flexibility**: We can swap implementations without changing RAGService

**Alex**: So for testing, we could inject a `MockDocumentRepository` that doesn't actually use ChromaDB?

**Sarah**: Exactly! You're getting the patterns now.

**Alex**: What's `@st.cache_resource`?

**Sarah**: Streamlit decorator. It ensures this function runs only once and caches the result. Otherwise, Streamlit would recreate everything on each interaction.

**Alex**: OK, and the UI has two panels?

**Sarah**: Yep! Look at the main function:

```python
col1, col2 = st.columns([1, 2])

with col1:
    document_panel(rag_service)

with col2:
    chat_panel(chat_service)
```

**Alex**: A 1:2 ratio, so the chat panel is twice as wide?

**Sarah**: Exactly! The `document_panel` handles uploads, the `chat_panel` handles conversation.

**Alex**: In `chat_panel`, there's this:

```python
for chunk in chat_service.send_message_stream(prompt):
    if hasattr(chunk, "content"):
        # Final message
        sources = chunk.sources
    else:
        # Text chunk
        full_response += chunk
        message_placeholder.markdown(full_response + "▌")
```

**Sarah**: Good eye! The service yields text chunks as they come in. We append them to `full_response` and update the placeholder. The "▌" is a blinking cursor effect.

**Alex**: Oh! So the user sees the response being typed out in real-time?

**Sarah**: Exactly! Then at the end, we get the final `Message` object with sources.

## Document Loading

**Alex**: How do PDFs get processed?

**Sarah**: Check `infrastructure/document_loaders/document_loader.py`:

```python
@staticmethod
def load_from_pdf(file_bytes: bytes, file_name: str) -> Document:
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))

    text_content = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text.strip():
            text_content.append(text)

    content = "\n\n".join(text_content)

    return Document(
        content=content,
        source_type="pdf",
        file_name=file_name,
        metadata={"page_count": len(pdf_reader.pages)},
    )
```

**Alex**: It reads each page, extracts text, joins them, and creates a Document entity?

**Sarah**: Yep! Notice it returns a `Document`, not some PDF-specific object. Once loaded, the system doesn't care if it was originally a PDF or text file.

**Alex**: That's the beauty of the Entity pattern - everything becomes a Document!

**Sarah**: Exactly! Now you're thinking in domain concepts.

## Putting It All Together

**Sarah**: Let's trace a full user interaction. User uploads a PDF:

**Alex**: OK, let me try:
1. UI calls `DocumentLoader.load_from_pdf`
2. Creates a `Document` entity
3. UI calls `rag_service.add_document`
4. RAGService calls `repository.save`
5. ChromaDocumentRepository chunks it and stores in ChromaDB

**Sarah**: Perfect! Now user asks a question:

**Alex**:
1. UI calls `chat_service.send_message_stream`
2. ChatService creates a user `Message`, adds to session
3. Calls `rag_service.query_stream`
4. RAGService calls `repository.search_similar`
5. ChromaDB returns similar chunks
6. RAGService builds context from chunks
7. Calls `openai_service.generate_response_stream` with context
8. OpenAI streams back response chunks
9. ChatService yields them to UI
10. UI displays in real-time
11. Finally, creates assistant `Message` with sources

**Sarah**: Absolutely perfect! You've got the full flow!

## What About Error Handling?

**Alex**: I notice there are try-catch blocks in the UI, but not many in the services. Why?

**Sarah**: Great observation! In the domain and application layers, we let exceptions bubble up. The presentation layer (UI) catches them and displays user-friendly messages.

**Alex**: So errors like "document validation failed" come from the Entity, but the UI catches and shows "Invalid document: content cannot be empty"?

**Sarah**: Exactly! This is separation of concerns. The domain enforces rules, the UI communicates them to users.

**Alex**: What about in production?

**Sarah**: You'd add more robust error handling:
- Retry logic for API calls
- Logging at each layer
- Monitoring and alerting
- Graceful degradation

## Testing

**Alex**: How would you test this?

**Sarah**: Great question! Remember the repository pattern?

```python
def test_rag_query():
    # Create mock repository
    mock_repo = Mock(spec=IDocumentRepository)
    mock_repo.search_similar.return_value = [
        (Document(content="Test"), 0.8)
    ]

    # Create mock LLM
    mock_llm = Mock(spec=OpenAIService)
    mock_llm.generate_response.return_value = "Answer"

    # Inject mocks
    rag_service = RAGService(mock_repo, mock_llm)

    # Test
    response, sources = rag_service.query("test")

    assert response == "Answer"
    mock_repo.search_similar.assert_called_once()
```

**Alex**: Oh! Because RAGService depends on interfaces, we can inject fakes!

**Sarah**: Exactly! This is why Dependency Injection is so powerful for testing.

## Final Thoughts

**Sarah**: So, what do you think?

**Alex**: Honestly, at first it looked complicated, but breaking it down layer by layer makes so much sense. Each piece has a clear job:
- Entities know their rules
- Repositories abstract storage
- Services orchestrate workflows
- Infrastructure handles external systems
- UI manages user interaction

**Sarah**: Perfect summary! And the best part?

**Alex**: I can swap ChromaDB for Pinecone, or OpenAI for Claude, and only change infrastructure files?

**Sarah**: Bingo! That's clean architecture in action.

**Alex**: One last question - why go through all this? Why not just put everything in one file?

**Sarah**: Great question. For a tiny project, one file might be fine. But as you grow:
- **Maintainability**: Clear structure makes changes easier
- **Testing**: Isolated components are testable
- **Team work**: Multiple developers can work on different layers
- **Evolution**: Swap implementations without breaking everything

**Alex**: So it's an investment in the future?

**Sarah**: Exactly! Like building a house with a good foundation. Costs more upfront, but saves you later.

**Alex**: This has been super helpful. Can I set up a session next week to build a feature together?

**Sarah**: Absolutely! What feature?

**Alex**: Maybe adding Word document support? Seems like I'd just need to:
1. Add a loader in `DocumentLoader`
2. Update the UI to accept .docx files
3. Everything else stays the same?

**Sarah**: Perfect! You've totally got it. The architecture makes features like that easy to add.

**Alex**: Thanks so much, Sarah!

**Sarah**: Anytime! Happy coding!

---

**End of Session**

## Key Takeaways (Alex's Notes)

- **Layered Architecture**: Each layer has a purpose
- **Domain Entities**: Enforce their own rules (invariants)
- **Repository Pattern**: Abstract storage behind interfaces (anti-corruption layer)
- **Dependency Injection**: Pass dependencies in, don't create them
- **RAG Pattern**: Retrieve → Augment → Generate
- **Chunking**: Break documents into searchable pieces
- **Embeddings**: Represent text as numbers that capture meaning
- **Streaming**: Yield response chunks for better UX
- **Clean Architecture**: Makes testing and evolution easier
