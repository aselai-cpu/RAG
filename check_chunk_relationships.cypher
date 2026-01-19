// Check Chunk Relationships in Neo4j
// Run this in Neo4j Browser (http://localhost:7474)

// 1. Count total chunks
MATCH (c:Chunk)
RETURN count(c) as total_chunks;

// 2. Count chunks with entities
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN count(DISTINCT c) as chunks_with_entities;

// 3. Count total entities
MATCH (e:Entity)
RETURN count(e) as total_entities;

// 4. Check SHARES_ENTITY relationships
MATCH ()-[r:SHARES_ENTITY]->()
RETURN count(r) as shares_entity_count;

// 5. Show sample SHARES_ENTITY relationships
MATCH (c1:Chunk)-[r:SHARES_ENTITY]->(c2:Chunk)
RETURN c1.id as chunk1_id, 
       c2.id as chunk2_id,
       r.shared_entity_count as shared_count,
       c1.document_id as doc1_id,
       c2.document_id as doc2_id,
       CASE WHEN c1.document_id = c2.document_id THEN 'same doc' ELSE 'cross-doc' END as relationship_type
LIMIT 20;

// 6. Check CO_OCCURS_WITH relationships
MATCH ()-[r:CO_OCCURS_WITH]->()
RETURN count(r) as co_occurs_count;

// 7. Check MENTIONS relationships (chunk to entity)
MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
RETURN count(r) as mentions_count;

// 8. Find chunks that SHOULD have relationships but don't
// (chunks that share entities but don't have SHARES_ENTITY relationship)
MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
WHERE c1.id <> c2.id
  AND NOT (c1)-[:SHARES_ENTITY]-(c2)
WITH c1, c2, count(DISTINCT e) as shared_entities
WHERE shared_entities >= 1
RETURN c1.id as chunk1_id,
       c2.id as chunk2_id,
       shared_entities,
       c1.document_id as doc1_id,
       c2.document_id as doc2_id
LIMIT 20;

// 9. Show entity distribution (which entities are mentioned in most chunks)
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WITH e, count(DISTINCT c) as chunk_count
ORDER BY chunk_count DESC
LIMIT 10
RETURN e.name as entity_name, 
       e.type as entity_type, 
       chunk_count;

// 10. Show chunks with most entities
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WITH c, count(e) as entity_count
ORDER BY entity_count DESC
LIMIT 10
RETURN c.id as chunk_id,
       c.document_id as document_id,
       entity_count;
