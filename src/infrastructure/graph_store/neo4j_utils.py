"""
Neo4j Utility Functions - Helper functions for debugging and verification.
"""
from src.infrastructure.graph_store.neo4j_repository import Neo4jRepository
from src.infrastructure.logging import RAGLogger

logger = RAGLogger.get_logger('neo4j_utils')


def verify_chunk_relationships(neo4j_repo: Neo4jRepository) -> dict:
    """
    Verify and report on chunk relationships in Neo4j.
    
    Args:
        neo4j_repo: Neo4j repository instance
        
    Returns:
        Dictionary with statistics about chunk relationships
    """
    with neo4j_repo.driver.session() as session:
        # Count chunks
        chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) as count").single()["count"]
        
        # Count chunks with entities
        chunks_with_entities = session.run(
            "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN count(DISTINCT c) as count"
        ).single()["count"]
        
        # Count SHARES_ENTITY relationships
        shares_entity_count = session.run(
            "MATCH ()-[r:SHARES_ENTITY]->() RETURN count(r) as count"
        ).single()["count"]
        
        # Count CO_OCCURS_WITH relationships
        co_occurs_count = session.run(
            "MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) as count"
        ).single()["count"]
        
        # Get sample relationships
        sample_relationships = session.run(
            """
            MATCH (c1:Chunk)-[r:SHARES_ENTITY]->(c2:Chunk)
            RETURN c1.id as chunk1_id, c2.id as chunk2_id, 
                   r.shared_entity_count as shared_count,
                   c1.document_id as doc1_id, c2.document_id as doc2_id
            LIMIT 10
            """
        )
        
        samples = []
        for record in sample_relationships:
            samples.append({
                "chunk1": record["chunk1_id"],
                "chunk2": record["chunk2_id"],
                "shared_entities": record["shared_count"],
                "same_document": record["doc1_id"] == record["doc2_id"],
            })
        
        stats = {
            "total_chunks": chunk_count,
            "chunks_with_entities": chunks_with_entities,
            "shares_entity_relationships": shares_entity_count,
            "co_occurs_relationships": co_occurs_count,
            "sample_relationships": samples,
        }
        
        logger.info(f"Chunk relationship stats: {stats}")
        return stats


def force_link_chunks_sharing_entities(neo4j_repo: Neo4jRepository, min_shared: int = 1):
    """
    Force link all chunks that share entities (useful for fixing missing relationships).
    
    Args:
        neo4j_repo: Neo4j repository instance
        min_shared: Minimum number of shared entities
    """
    with neo4j_repo.driver.session() as session:
        result = session.run(
            """
            MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
            WHERE c1.id <> c2.id
            WITH c1, c2, count(DISTINCT e) as shared_entity_count
            WHERE shared_entity_count >= $min_shared
            MERGE (c1)-[r:SHARES_ENTITY]->(c2)
            SET r.shared_entity_count = shared_entity_count,
                r.weight = shared_entity_count,
                r.created_at = datetime()
            RETURN count(r) as created_count
            """,
            min_shared=min_shared,
        )
        
        count = result.single()["created_count"]
        logger.info(f"Force-linked {count} chunk relationships")
        return count
