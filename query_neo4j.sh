#!/bin/bash

# Query Neo4j to check chunk relationships

echo "=========================================="
echo "Neo4j Chunk Relationships Check"
echo "=========================================="
echo ""

echo "1. Total chunks:"
echo "MATCH (c:Chunk) RETURN count(c);" | docker exec -i rag-neo4j cypher-shell -u neo4j -p password --format plain

echo ""
echo "2. Chunks with entities:"
echo "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN count(DISTINCT c);" | docker exec -i rag-neo4j cypher-shell -u neo4j -p password --format plain

echo ""
echo "3. SHARES_ENTITY relationships count:"
echo "MATCH ()-[r:SHARES_ENTITY]->() RETURN count(r);" | docker exec -i rag-neo4j cypher-shell -u neo4j -p password --format plain

echo ""
echo "4. Sample SHARES_ENTITY relationships:"
echo "MATCH (c1:Chunk)-[r:SHARES_ENTITY]->(c2:Chunk) RETURN c1.id, c2.id, r.shared_entity_count LIMIT 10;" | docker exec -i rag-neo4j cypher-shell -u neo4j -p password --format plain

echo ""
echo "5. Chunks that share entities but don't have SHARES_ENTITY relationship:"
echo "MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk) WHERE c1.id <> c2.id AND NOT (c1)-[:SHARES_ENTITY]-(c2) WITH c1, c2, count(DISTINCT e) as shared WHERE shared >= 1 RETURN count(*) as missing;" | docker exec -i rag-neo4j cypher-shell -u neo4j -p password --format plain

echo ""
echo "=========================================="
