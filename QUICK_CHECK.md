# Quick Check: Chunk Relationships in Neo4j

## Option 1: Using Neo4j Browser (Easiest)

1. Open Neo4j Browser: http://localhost:7474
2. Login with credentials: `neo4j` / `password`
3. Copy and paste the queries from `check_chunk_relationships.cypher` one by one

## Option 2: Using cypher-shell (Command Line)

If you have `cypher-shell` installed (comes with Neo4j):

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Run queries
docker exec -it rag-neo4j cypher-shell -u neo4j -p password < check_chunk_relationships.cypher
```

## Option 3: Quick Single Query

Run this in Neo4j Browser to see all chunk relationships:

```cypher
// Quick overview
MATCH (c1:Chunk)-[r:SHARES_ENTITY]->(c2:Chunk)
RETURN c1.id, c2.id, r.shared_entity_count, 
       c1.document_id as doc1, c2.document_id as doc2
LIMIT 50;
```

## Option 4: Visual Graph View

In Neo4j Browser, run:

```cypher
MATCH (c1:Chunk)-[r:SHARES_ENTITY]->(c2:Chunk)
RETURN c1, r, c2
LIMIT 50;
```

Then click the "Graph" view to see the visual representation.

## What to Look For

- **SHARES_ENTITY relationships**: Should show connections between chunks
- **shared_entity_count**: Number of entities shared between chunks
- **same doc vs cross-doc**: Whether chunks are from same document or different documents

## If No Relationships Found

1. Check if chunks have entities:
   ```cypher
   MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
   RETURN c.id, e.name
   LIMIT 20;
   ```

2. Check if chunks share entities (potential relationships):
   ```cypher
   MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
   WHERE c1.id <> c2.id
   RETURN c1.id, c2.id, count(DISTINCT e) as shared
   LIMIT 20;
   ```

3. If chunks share entities but no relationships exist, run the diagnostic script:
   ```bash
   python3 diagnose_chunk_relationships.py
   ```
