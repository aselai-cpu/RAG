# Frequently Asked Questions (FAQ)

## General Questions

### What is RAG?

**RAG stands for Retrieval-Augmented Generation**. It's a technique that combines:
- **Retrieval**: Finding relevant information from your documents
- **Generation**: Using AI to create answers based on that information

Think of it as giving an AI assistant access to your personal library.

### Why use RAG instead of fine-tuning an LLM?

**RAG Advantages**:
- Update information instantly (just upload new documents)
- No expensive retraining required
- Works with any LLM
- Transparent (shows sources)
- Cost-effective for frequently changing information

**Fine-tuning Advantages**:
- Better for teaching style/tone
- Can encode large amounts of domain knowledge
- No retrieval latency

**When to use RAG**: Dynamic information, fact-based Q&A, document analysis
**When to fine-tune**: Specialized language style, domain-specific tasks, when you have large training budgets

### How is this different from ChatGPT?

ChatGPT knows general knowledge but not YOUR documents. This application:
1. Stores your documents
2. Searches them when you ask questions
3. Provides answers based on YOUR content
4. Shows which documents were used

### Do I need coding experience to use this?

**To use the application**: No programming knowledge needed. Just upload documents and ask questions.

**To modify or extend**: Basic Python knowledge is helpful, but the code is well-documented for learning.

### Is my data private?

**What stays local**:
- Your uploaded documents
- Document embeddings
- Chat history

**What goes to OpenAI**:
- Your questions
- Retrieved context chunks (not full documents)
- Chat history for context

**Note**: OpenAI's API doesn't train on your data per their usage policy.

## Technical Questions

### What vector database does this use?

**ChromaDB** - a lightweight, Python-native vector database perfect for learning and prototypes.

**Why ChromaDB?**
- Easy to set up (no separate server)
- Works locally
- Good performance for small-medium datasets
- Python-native API

**Alternatives**: Pinecone, Weaviate, Qdrant, FAISS

### How does text chunking work?

Documents are split into chunks using `RecursiveCharacterTextSplitter`:

**Configuration**:
- **Chunk size**: 1000 characters
- **Overlap**: 200 characters
- **Separators**: Prioritizes paragraph breaks, then sentences, then words

**Why overlap?** Prevents important information from being split across chunks.

**Example**:
```
Document: "...end of chunk 1. Important information here. Start of chunk 2..."

Chunk 1: "...end of chunk 1. Important information here."
Chunk 2: "Important information here. Start of chunk 2..."  ← Overlap
```

### What embedding model is used?

Currently using ChromaDB's default embedding function.

**For production**, explicitly specify:
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### How many documents can it handle?

**Current setup** (local ChromaDB):
- **Optimal**: Up to 100,000 chunks (~500-1000 documents)
- **Functional**: Up to 1 million chunks (slower)

**Scalability options**:
- ChromaDB Cloud for larger deployments
- Switch to Pinecone or Weaviate for production scale

### What file types are supported?

Currently:
- **PDF** (.pdf)
- **Text files** (.txt)
- **Direct text input** (paste/clipboard)

**Easy to add**:
- Word documents (.docx) - use `python-docx`
- Markdown (.md) - just text
- HTML (.html) - use `BeautifulSoup`

### How do I change the chunk size?

In `chroma_document_repository.py`, modify:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Change this
    chunk_overlap=200,     # And this
    # ...
)
```

**Guidelines**:
- **Smaller chunks** (500): More precise retrieval, but less context
- **Larger chunks** (2000): More context, but less precise
- **Overlap**: Usually 10-20% of chunk size

### Can I use a different LLM?

Yes! The code uses OpenAI, but you can swap in any LLM.

**Example - Anthropic Claude**:
```python
from anthropic import Anthropic

class ClaudeService:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)

    def generate_response(self, messages, context=None):
        # Implement Claude API call
        response = self.client.messages.create(...)
        return response.content
```

**Example - Local LLM (Ollama)**:
```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")
```

## Usage Questions

### How do I get the best answers?

**1. Ask specific questions**:
- ✅ Good: "What is the vacation policy for new employees?"
- ❌ Poor: "Tell me about policies"

**2. Use keywords from your documents**:
- If your documents use "PTO", use "PTO" in questions

**3. One concept per question**:
- ✅ Good: "What is the remote work policy?"
- ❌ Poor: "What are the remote work, vacation, and benefits policies?"

**4. Provide context if needed**:
- "According to the 2024 handbook, what is the remote work policy?"

### Why are some answers not accurate?

**Common Causes**:

1. **No relevant documents**: The information isn't in your uploaded documents
2. **Poor document quality**: Scanned PDFs without OCR, corrupted files
3. **Ambiguous questions**: Question is too vague or broad
4. **Similarity threshold**: Relevant chunks scored below 0.5 threshold
5. **LLM limitations**: Even with context, LLM might misinterpret

**Solutions**:
- Upload more comprehensive documents
- Ask more specific questions
- Lower similarity threshold if needed
- Check retrieved sources to verify

### Can I see what documents were used for an answer?

Yes! Each response shows sources in an expandable section:
- Click "📎 Sources" to view
- Shows document IDs and relevance scores

### How do I delete documents?

In the left panel:
1. Find the document in the list
2. Click the trash icon (🗑️)
3. Document and all its chunks are removed

### Can I have multiple conversations?

Currently, the app has a single conversation session.

**To add multiple conversations**:
Would require:
- Session management in the UI
- Multiple ChatSession entities
- Session persistence

This is a great feature to add!

### How do I export my chat history?

Currently not built-in, but easy to add:

```python
# Add to chat panel
if st.button("Export Chat"):
    chat_json = json.dumps([msg for msg in st.session_state.messages], indent=2)
    st.download_button("Download", chat_json, "chat_history.json")
```

## Performance Questions

### Why is the first query slow?

**Reasons**:
1. **Model loading**: ChromaDB loads embedding model
2. **Index building**: First query might trigger index optimization
3. **Cold start**: Python/Streamlit initialization

**Subsequent queries** are much faster (typically <2 seconds).

### How can I speed up document processing?

**Current**: Documents processed sequentially

**Optimization**:
```python
# Batch processing
def add_documents_batch(self, documents: List[Document]):
    all_chunks = []
    for doc in documents:
        chunks = self.text_splitter.split_text(doc.content)
        all_chunks.extend(chunks)

    # Single batch insert
    self.collection.add(documents=all_chunks, ...)
```

### Why does the UI freeze during queries?

**Reason**: Streamlit reruns entire script on each interaction.

**Solution**: Already implemented - streaming responses provide real-time feedback.

**Further optimization**: Use `@st.cache_resource` for expensive operations.

### How much does OpenAI API cost?

**Approximate costs** (as of 2024):

**Embeddings** (if using OpenAI embeddings):
- text-embedding-3-small: $0.02 per 1M tokens
- text-embedding-3-large: $0.13 per 1M tokens

**Chat Completions**:
- GPT-4 Turbo: $0.01 per 1K input tokens, $0.03 per 1K output tokens
- GPT-3.5 Turbo: $0.0005 per 1K input tokens, $0.0015 per 1K output tokens

**Example**: 100 documents + 100 queries ≈ $1-5 depending on document size

**Cost reduction**:
- Use GPT-3.5 Turbo instead of GPT-4
- Use smaller embedding model
- Cache common queries

## Troubleshooting

### "OpenAI API key not found"

**Solution**:
1. Copy `.env.example` to `.env`
2. Add your key: `OPENAI_API_KEY=sk-your-key-here`
3. Restart the application

### "Module not found" error

**Solution**:
```bash
pip install -r requirements.txt
```

If still fails:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### PDF not uploading

**Possible causes**:
1. **Scanned PDF**: No text layer (use OCR first)
2. **Corrupted file**: Try opening in PDF reader
3. **File too large**: Max 10MB by default
4. **Protected PDF**: Remove password protection first

**Solution for scanned PDFs**:
Use OCR tools:
- Adobe Acrobat (paid)
- OCRmyPDF (free)
- Cloud services (Google Drive, Adobe online)

### ChromaDB errors

**"Collection already exists"**:
```python
# Use get_or_create instead of create
collection = client.get_or_create_collection(name="rag_documents")
```

**"Dimension mismatch"**:
- Embeddings model changed
- Delete `./data/chroma` directory and re-upload documents

**Database locked**:
- Another instance running
- Close other instances or delete lock file

### Streamlit issues

**App not opening**:
```bash
# Try different port
streamlit run src/presentation/ui/app.py --server.port=8502
```

**Session state reset**:
- This is normal Streamlit behavior on refresh
- For persistent state, implement file-based storage

## Deployment Questions

### Can I deploy this to production?

Yes, but consider:

**Required enhancements**:
- [ ] User authentication
- [ ] Multi-user support
- [ ] Persistent session storage
- [ ] Rate limiting
- [ ] Error monitoring
- [ ] Backup strategy

**Deployment options**:
- Streamlit Cloud (easiest)
- Docker container on any cloud (AWS, GCP, Azure)
- Kubernetes for scale

### How do I dockerize this?

Dockerfile is provided. Run:
```bash
docker build -t rag-app .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-your-key rag-app
```

### What about authentication?

Add Streamlit authentication:
```python
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    cookie_name,
    key,
    cookie_expiry_days
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show app
    main()
```

Or use OAuth providers (Google, GitHub) via Streamlit Cloud.

## Development Questions

### How do I contribute?

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit pull request

**Areas needing contribution**:
- Additional document loaders (Word, HTML)
- Advanced retrieval methods (hybrid search)
- UI improvements
- Performance optimizations
- More tests

### How do I run tests?

```bash
pytest tests/
```

**Note**: Tests are not included in the base project. Add them following the testing strategy in the professional guide.

### Can I use this commercially?

Yes! This project is provided for both educational and commercial use.

**Disclaimer**: Review OpenAI's terms of service for commercial API use.

### How do I add new features?

Follow the DDD architecture:

1. **New entity**: Add to `domain/entities/`
2. **New repository**: Interface in `domain/repositories/`, implementation in `infrastructure/`
3. **New service**: Add to `application/services/`
4. **UI component**: Add to `presentation/`

**Example - Add document tagging**:
```python
# 1. Update entity
class Document:
    tags: List[str] = field(default_factory=list)

# 2. Update repository
class IDocumentRepository:
    def search_by_tag(self, tag: str) -> List[Document]:
        pass

# 3. Implement in infrastructure
class ChromaDocumentRepository:
    def search_by_tag(self, tag: str) -> List[Document]:
        return self.collection.get(where={"tag": tag})

# 4. Add UI in app.py
selected_tag = st.selectbox("Filter by tag", tags)
filtered_docs = rag_service.get_documents_by_tag(selected_tag)
```

## Best Practices

### Document Organization

**Do**:
- Use clear, descriptive file names
- Organize by topic/category
- Keep documents updated
- Remove duplicates

**Don't**:
- Upload everything hoping for magic
- Mix unrelated content in single documents
- Keep outdated versions

### Query Formulation

**Do**:
- Be specific
- Use terminology from your documents
- Ask one question at a time
- Provide context if ambiguous

**Don't**:
- Ask overly broad questions
- Expect answers not in your documents
- Use vague pronouns without context

### System Maintenance

**Do**:
- Regularly update documents
- Monitor query quality
- Review and clean up old documents
- Keep dependencies updated

**Don't**:
- Let database grow unbounded
- Ignore slow queries
- Skip backups

## Getting Help

### Documentation

1. **Novice Guide**: Basic concepts and usage
2. **Professional Guide**: Technical deep dive
3. **Philosophical Foundation**: Design rationale
4. **Architecture Docs**: System design
5. **Code Walkthrough**: Line-by-line explanation

### Community Resources

- **LangChain Documentation**: https://python.langchain.com/
- **ChromaDB Documentation**: https://docs.trychroma.com/
- **OpenAI API Reference**: https://platform.openai.com/docs/

### Support Channels

For bugs or feature requests:
- Open an issue on GitHub
- Include error messages and steps to reproduce
- Provide system information (Python version, OS)

---

**Didn't find your question?** Check the other documentation or open an issue!
