# Graph RAG Setup Guide

This guide explains how to set up and use the Graph RAG system that combines ChromaDB vector search with Neo4j graph traversal.

## Overview

The Graph RAG system enhances traditional RAG by:
1. **Vector Search (ChromaDB)**: Finds semantically similar chunks to the query
2. **Graph Enrichment (Neo4j)**: Finds related chunks through entity relationships
3. **Hybrid Retrieval**: Combines both approaches for richer context

## Architecture

### Graph Schema

The Neo4j graph stores:
- **Nodes**:
  - `Document`: Represents uploaded documents
  - `Chunk`: Represents text chunks from documents
  - `Entity`: Extracted entities (Person, Organization, Concept, Location, etc.)

- **Relationships**:
  - `(Document)-[:CONTAINS]->(Chunk)`: Document contains chunks
  - `(Chunk)-[:MENTIONS]->(Entity)`: Chunk mentions entities
  - `(Entity)-[:RELATED_TO]->(Entity)`: Entities are related
  - `(Chunk)-[:CO_OCCURS_WITH]->(Chunk)`: Chunks co-occur

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `neo4j==5.20.0`: Neo4j Python driver
- All other existing dependencies

### 2. Start Neo4j with Docker

```bash
docker-compose up -d
```

This will start Neo4j on:
- **HTTP**: http://localhost:7474 (Neo4j Browser)
- **Bolt**: bolt://localhost:7687 (Application connection)

Default credentials:
- Username: `neo4j`
- Password: `password`

### 3. Configure Environment Variables (Optional)

You can customize Neo4j connection via environment variables:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

If not set, defaults will be used.

### 4. Run the Application

```bash
streamlit run src/presentation/ui/app.py
```

The application will:
- Automatically detect if Neo4j is available
- Enable Graph RAG if Neo4j is running
- Fall back to vector-only retrieval if Neo4j is unavailable

## How It Works

### Document Ingestion

When you upload a document:

1. **Vector Storage (ChromaDB)**:
   - Document is chunked
   - Chunks are embedded and stored in ChromaDB

2. **Graph Storage (Neo4j)**:
   - Entities are extracted from each chunk using LLM
   - Relationships between entities are identified
   - Graph nodes and relationships are created in Neo4j
   - Chunks are linked to entities they mention

### Query Processing

When you ask a question:

1. **Vector Search**:
   - Query is embedded
   - Similar chunks are retrieved from ChromaDB

2. **Graph Enrichment**:
   - Entities are extracted from the query
   - Related chunks are found through:
     - Chunks mentioning the same entities
     - Chunks connected through entity relationships
     - Chunks that co-occur with retrieved chunks

3. **Hybrid Results**:
   - Vector results and graph results are combined
   - Duplicates are removed
   - Results are ranked by relevance

4. **Response Generation**:
   - Enriched context is sent to LLM
   - Response includes sources from both vector and graph

## Features

### Dynamic Ontology

The system builds an ontology dynamically based on document content:
- Entity types are inferred from context
- Relationships are discovered automatically
- No predefined schema required

### Entity Types

The system extracts various entity types:
- **Person**: People mentioned in documents
- **Organization**: Companies, institutions, groups
- **Concept**: Ideas, topics, themes
- **Location**: Places, geographic locations
- **Event**: Events, occurrences
- **Product**: Products, services
- **Technology**: Technologies, tools, systems
- **Other**: Any other relevant entities

### Relationship Types

Common relationship types include:
- `RELATED_TO`: General relationship
- `WORKS_FOR`: Employment relationship
- `LOCATED_IN`: Location relationship
- `PART_OF`: Membership/containment
- `CREATED_BY`: Creation relationship
- And more based on content

## UI Features

### Source Attribution

Sources are labeled with their origin:
- 📊 **Vector**: Retrieved via semantic similarity
- 🕸️ **Graph**: Retrieved via graph traversal

### Statistics

The sidebar shows:
- **ChromaDB**: Total chunks and documents
- **Neo4j Graph**: Documents, chunks, entities, and relationships

## Troubleshooting

### Neo4j Not Available

If you see a warning about Neo4j:
1. Check if Docker is running: `docker ps`
2. Start Neo4j: `docker-compose up -d`
3. Verify connection: Visit http://localhost:7474

### Entity Extraction Issues

If entities aren't being extracted:
- Check OpenAI API key is set
- Review logs in `logs/` directory
- Ensure documents contain named entities

### Performance

For large document sets:
- Entity extraction can be slow (uses LLM per chunk)
- Consider processing in batches
- Monitor Neo4j memory usage

## Advanced Usage

### Accessing Neo4j Browser

1. Visit http://localhost:7474
2. Login with credentials (default: neo4j/password)
3. Run Cypher queries to explore the graph

Example queries:

```cypher
// View all entities
MATCH (e:Entity)
RETURN e.name, e.type
LIMIT 20

// View document structure
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN d.file_name, count(DISTINCT c) as chunks, count(DISTINCT e) as entities

// Find related entities
MATCH (e1:Entity)-[:RELATED_TO]->(e2:Entity)
RETURN e1.name, e2.name, type(rel) as relationship
LIMIT 20
```

### Customizing Entity Extraction

Modify `src/application/services/entity_extraction_service.py` to:
- Change entity types
- Adjust extraction prompts
- Modify relationship types

### Tuning Hybrid Retrieval

Modify `src/application/services/hybrid_retrieval_service.py` to:
- Adjust `vector_top_k` and `graph_top_k`
- Change graph traversal depth
- Modify relevance scoring

## Architecture Notes

The system follows clean architecture principles:
- **Domain Layer**: Core entities and interfaces
- **Application Layer**: Business logic (RAG, entity extraction, hybrid retrieval)
- **Infrastructure Layer**: External services (ChromaDB, Neo4j, OpenAI)

This allows easy swapping of components and testing.

## Next Steps

- Experiment with different entity types
- Explore the graph in Neo4j Browser
- Monitor retrieval quality
- Adjust parameters based on your use case
