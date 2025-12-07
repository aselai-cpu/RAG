# RAG Application Logs

This directory contains comprehensive logs for the RAG application, tracking user actions, system operations, and inter-service communication.

## Log Files

### 1. `rag_app.log`
**Main application log** - General application events and operations

**Contents:**
- Application initialization
- Service creation and configuration
- General system events
- Performance metrics

**Example entries:**
```
2025-12-07 12:00:00 | rag_app | INFO | setup_logging:75 | RAG Application Logging Initialized
2025-12-07 12:00:05 | rag_app | INFO | initialize_services:87 | Services initialized successfully
```

### 2. `user_actions.log`
**User activity log** - All user interactions with the application

**Contents:**
- Document uploads (files and pasted text)
- Chat queries submitted
- Button clicks (upload, clear chat, etc.)
- User errors and warnings

**Example entries:**
```
2025-12-07 12:01:15 | INFO | User clicked 'Upload Document' button: filename=policy.pdf, size=245632 bytes
2025-12-07 12:01:17 | INFO | Document uploaded successfully: doc_id=abc123, filename=policy.pdf, chunks=12
2025-12-07 12:02:30 | INFO | User submitted question: query_length=35, query='What is the vacation policy?'
```

### 3. `chromadb.log`
**Vector database operations log** - All ChromaDB interactions

**Contents:**
- Document save operations with chunking details
- Similarity search queries with results
- Database operations (get, delete)
- Performance timing for each operation

**Example entries:**
```
2025-12-07 12:01:16 | chromadb_ops | INFO | save:77 | Starting document save: doc_id=abc123, source_type=pdf, file_name=policy.pdf
2025-12-07 12:01:16 | chromadb_ops | DEBUG | save:83 | Document chunked: doc_id=abc123, chunk_count=12, content_length=8456
2025-12-07 12:01:17 | chromadb_ops | INFO | save:111 | Document saved successfully: doc_id=abc123, chunks=12, time=0.34s
2025-12-07 12:02:30 | chromadb_ops | INFO | search_similar:215 | Starting similarity search: query_length=35, top_k=5
2025-12-07 12:02:30 | chromadb_ops | INFO | search_similar:255 | Search completed: num_results=5, time=0.12s
```

### 4. `openai.log`
**OpenAI API calls log** - All LLM API interactions

**Contents:**
- API requests with model and parameters
- User queries sent to OpenAI
- Response details
- Token usage (prompt tokens, completion tokens, total)
- API call timing
- API errors

**Example entries:**
```
2025-12-07 12:02:30 | openai_api | INFO | generate_response:72 | Starting OpenAI API call: model=gpt-4-turbo-preview, temperature=0.7, num_messages=1, has_context=True
2025-12-07 12:02:30 | openai_api | INFO | generate_response:107 | User query: 'What is the vacation policy?'
2025-12-07 12:02:32 | openai_api | INFO | generate_response:122 | OpenAI API usage: prompt_tokens=456, completion_tokens=89, total_tokens=545
2025-12-07 12:02:32 | openai_api | INFO | generate_response:127 | OpenAI API call completed: response_length=348, time=1.95s
```

### 5. `errors.log`
**Error log** - All errors and exceptions

**Contents:**
- Application errors
- User-triggered errors (invalid files, etc.)
- System errors
- Stack traces for debugging

**Example entries:**
```
2025-12-07 12:05:15 | errors | ERROR | Error uploading document: filename=corrupted.pdf, error=Failed to read PDF
Traceback (most recent call last):
  File "...app.py", line 150, in upload_document
    document = DocumentLoader.load_from_pdf(file_bytes, filename)
  ...
```

## Log File Rotation

Logs are automatically rotated to prevent unlimited growth:
- **Max size per file**: 10MB (5MB for user_actions)
- **Backup count**: 5 files (10 for errors)
- **Naming**: `rag_app.log`, `rag_app.log.1`, `rag_app.log.2`, etc.

When a log file reaches its max size, it's renamed with a number suffix and a new file is created.

## Log Levels

The application uses different log levels:
- **DEBUG**: Detailed information for diagnosing problems (verbose)
- **INFO**: General informational messages about normal operation
- **WARNING**: Warnings about potential issues
- **ERROR**: Errors that prevented an operation from completing

## Reading Logs

### View most recent entries
```bash
tail -f logs/user_actions.log
```

### Search for specific user action
```bash
grep "User submitted question" logs/user_actions.log
```

### Find all errors today
```bash
grep "$(date +%Y-%m-%d)" logs/errors.log
```

### Count API calls
```bash
grep "Starting OpenAI API call" logs/openai.log | wc -l
```

### Calculate total tokens used
```bash
grep "total_tokens" logs/openai.log | awk -F'total_tokens=' '{sum+=$2} END {print sum}'
```

### View document uploads
```bash
grep "Document uploaded successfully" logs/user_actions.log
```

## Use Cases

### 1. Debugging User Issues
Check `user_actions.log` to see what the user did, then check `errors.log` for any errors.

### 2. Monitoring API Usage
Check `openai.log` for token usage to estimate costs:
```bash
grep "OpenAI API usage" logs/openai.log | tail -20
```

### 3. Performance Monitoring
Check timing in `chromadb.log` and `openai.log`:
```bash
grep "time=" logs/chromadb.log | tail -10
grep "time=" logs/openai.log | tail -10
```

### 4. Understanding System Flow
Follow a single query through all logs:
1. User submits question (`user_actions.log`)
2. System searches ChromaDB (`chromadb.log`)
3. Context sent to OpenAI (`openai.log`)
4. Response generated (`openai.log`)
5. Response displayed to user (`user_actions.log`)

## Privacy and Security

**⚠️ Important**: Log files may contain:
- User queries (personal/confidential information)
- Document content excerpts
- OpenAI API requests and responses

**Recommendations**:
- Do NOT commit logs to version control (already in `.gitignore`)
- Restrict access to log directory in production
- Implement log retention policies
- Consider encrypting sensitive logs
- Scrub sensitive data before sharing logs

## Maintenance

### Clear old logs
```bash
# Keep only last 7 days
find logs/ -name "*.log*" -mtime +7 -delete
```

### Archive logs
```bash
# Archive logs older than 30 days
tar -czf logs-archive-$(date +%Y-%m-%d).tar.gz logs/
```

### Monitor disk usage
```bash
du -sh logs/
```

## Troubleshooting

### No logs being generated
1. Check file permissions: `ls -la logs/`
2. Check logger initialization in code
3. Verify logging module is imported

### Logs growing too large
1. Reduce log level from DEBUG to INFO
2. Decrease rotation maxBytes
3. Implement more aggressive rotation

### Missing some operations in logs
1. Check that logger is called in that code path
2. Verify log level allows that message type
3. Check for exceptions preventing logging

---

**Log files are essential for debugging, monitoring, and understanding your RAG application. Review them regularly!**
