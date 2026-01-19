#!/usr/bin/env python3
"""
Diagnostic script to check and fix chunk relationships in Neo4j.

This script helps diagnose why chunk relationships might not be appearing
and can force-create them if needed.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.infrastructure.graph_store.neo4j_repository import Neo4jRepository
from src.infrastructure.graph_store.neo4j_utils import verify_chunk_relationships, force_link_chunks_sharing_entities
from src.infrastructure.logging import RAGLogger

logger = RAGLogger.get_logger('diagnostics')


def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("Neo4j Chunk Relationships Diagnostic")
    print("=" * 80)
    print()
    
    try:
        # Initialize Neo4j repository
        print("Connecting to Neo4j...")
        neo4j_repo = Neo4jRepository()
        print("✓ Connected to Neo4j")
        print()
        
        # Verify current state
        print("Checking current state...")
        stats = verify_chunk_relationships(neo4j_repo)
        print()
        print("Current Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Chunks with entities: {stats['chunks_with_entities']}")
        print(f"  SHARES_ENTITY relationships: {stats['shares_entity_relationships']}")
        print(f"  CO_OCCURS_WITH relationships: {stats['co_occurs_relationships']}")
        print()
        
        if stats['sample_relationships']:
            print("Sample Relationships:")
            for rel in stats['sample_relationships'][:5]:
                doc_note = " (same doc)" if rel['same_document'] else " (cross-doc)"
                print(f"  {rel['chunk1']} <-> {rel['chunk2']} ({rel['shared_entities']} entities){doc_note}")
        else:
            print("⚠ No chunk relationships found!")
        print()
        
        # Check if chunks have entities
        if stats['chunks_with_entities'] == 0:
            print("❌ ERROR: No chunks have entities linked!")
            print("   This means entities weren't extracted or linked to chunks.")
            print("   Make sure you've uploaded documents and they were processed.")
            return
        
        if stats['chunks_with_entities'] < stats['total_chunks']:
            print(f"⚠ WARNING: Only {stats['chunks_with_entities']} of {stats['total_chunks']} chunks have entities")
            print()
        
        # Offer to fix
        if stats['shares_entity_relationships'] == 0:
            print("No SHARES_ENTITY relationships found.")
            response = input("Would you like to force-create relationships? (y/n): ")
            if response.lower() == 'y':
                print()
                print("Creating relationships...")
                count = force_link_chunks_sharing_entities(neo4j_repo, min_shared=1)
                print(f"✓ Created {count} relationships")
                print()
                
                # Verify again
                print("Verifying after fix...")
                stats = verify_chunk_relationships(neo4j_repo)
                print(f"  SHARES_ENTITY relationships: {stats['shares_entity_relationships']}")
        else:
            print("✓ Relationships exist. You can force-recreate them if needed.")
            response = input("Force-recreate all relationships? (y/n): ")
            if response.lower() == 'y':
                print()
                print("Recreating relationships...")
                count = force_link_chunks_sharing_entities(neo4j_repo, min_shared=1)
                print(f"✓ Created/updated {count} relationships")
        
        print()
        print("=" * 80)
        print("Diagnostic complete!")
        print()
        print("To view relationships in Neo4j Browser, run:")
        print("  MATCH (c1:Chunk)-[r:SHARES_ENTITY]->(c2:Chunk)")
        print("  RETURN c1.id, c2.id, r.shared_entity_count")
        print("  LIMIT 20")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
