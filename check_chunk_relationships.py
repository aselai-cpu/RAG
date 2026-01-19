#!/usr/bin/env python3
"""
Quick script to check chunk relationships in Neo4j.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.infrastructure.graph_store.neo4j_repository import Neo4jRepository

def main():
    print("=" * 80)
    print("Checking Neo4j Chunk Relationships")
    print("=" * 80)
    print()
    
    try:
        neo4j_repo = Neo4jRepository()
        
        with neo4j_repo.driver.session() as session:
            # 1. Count total chunks
            result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
            total_chunks = result.single()["count"]
            print(f"Total chunks in graph: {total_chunks}")
            
            # 2. Count chunks with entities
            result = session.run(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN count(DISTINCT c) as count"
            )
            chunks_with_entities = result.single()["count"]
            print(f"Chunks with entities: {chunks_with_entities}")
            
            # 3. Count total entities
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            total_entities = result.single()["count"]
            print(f"Total entities: {total_entities}")
            print()
            
            # 4. Check SHARES_ENTITY relationships
            print("SHARES_ENTITY Relationships:")
            print("-" * 80)
            result = session.run(
                """
                MATCH ()-[r:SHARES_ENTITY]->()
                RETURN count(r) as count
                """
            )
            shares_count = result.single()["count"]
            print(f"  Total SHARES_ENTITY relationships: {shares_count}")
            
            if shares_count > 0:
                # Get sample relationships
                result = session.run(
                    """
                    MATCH (c1:Chunk)-[r:SHARES_ENTITY]->(c2:Chunk)
                    RETURN c1.id as chunk1_id, 
                           c2.id as chunk2_id,
                           r.shared_entity_count as shared_count,
                           c1.document_id as doc1_id,
                           c2.document_id as doc2_id
                    LIMIT 10
                    """
                )
                print("\n  Sample relationships:")
                for record in result:
                    same_doc = "✓ same doc" if record["doc1_id"] == record["doc2_id"] else "✗ cross-doc"
                    print(f"    {record['chunk1_id']} <-> {record['chunk2_id']} "
                          f"({record['shared_count']} shared entities) [{same_doc}]")
            else:
                print("  ⚠ No SHARES_ENTITY relationships found!")
            print()
            
            # 5. Check CO_OCCURS_WITH relationships
            result = session.run(
                """
                MATCH ()-[r:CO_OCCURS_WITH]->()
                RETURN count(r) as count
                """
            )
            co_occurs_count = result.single()["count"]
            print(f"CO_OCCURS_WITH relationships: {co_occurs_count}")
            print()
            
            # 6. Check entity-to-chunk relationships
            result = session.run(
                """
                MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
                RETURN count(r) as count
                """
            )
            mentions_count = result.single()["count"]
            print(f"Chunk-to-Entity (MENTIONS) relationships: {mentions_count}")
            print()
            
            # 7. Check if chunks share entities (potential relationships)
            if chunks_with_entities > 1:
                result = session.run(
                    """
                    MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
                    WHERE c1.id <> c2.id
                    RETURN count(DISTINCT [c1.id, c2.id]) as potential_relationships
                    """
                )
                potential = result.single()["potential_relationships"]
                print(f"Potential chunk pairs that share entities: {potential}")
                if potential > shares_count:
                    print(f"  ⚠ {potential - shares_count} relationships could be created but aren't!")
            print()
            
            # 8. Show chunk-entity distribution
            print("Chunk-Entity Distribution:")
            print("-" * 80)
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WITH c, count(e) as entity_count
                RETURN min(entity_count) as min_entities,
                       max(entity_count) as max_entities,
                       avg(entity_count) as avg_entities,
                       count(c) as chunks_with_entities
                """
            )
            record = result.single()
            if record["chunks_with_entities"] > 0:
                print(f"  Chunks with entities: {record['chunks_with_entities']}")
                print(f"  Min entities per chunk: {record['min_entities']}")
                print(f"  Max entities per chunk: {record['max_entities']}")
                print(f"  Avg entities per chunk: {record['avg_entities']:.1f}")
            print()
            
            # 9. Show entity distribution across chunks
            print("Entity Distribution (top entities mentioned in most chunks):")
            print("-" * 80)
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WITH e, count(DISTINCT c) as chunk_count
                ORDER BY chunk_count DESC
                LIMIT 10
                RETURN e.name as entity_name, e.type as entity_type, chunk_count
                """
            )
            for record in result:
                print(f"  {record['entity_name']} ({record['entity_type']}): "
                      f"mentioned in {record['chunk_count']} chunks")
            print()
            
        print("=" * 80)
        print("Summary:")
        print(f"  - {total_chunks} chunks, {chunks_with_entities} with entities")
        print(f"  - {total_entities} entities")
        print(f"  - {shares_count} SHARES_ENTITY relationships")
        print(f"  - {co_occurs_count} CO_OCCURS_WITH relationships")
        print(f"  - {mentions_count} MENTIONS relationships")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
