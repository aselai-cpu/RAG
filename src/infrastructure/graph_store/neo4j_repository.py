"""
Neo4j Graph Repository - Infrastructure layer for graph-based storage.

This repository manages entities, relationships, and graph traversal
for Graph RAG functionality.
"""
from typing import List, Dict, Optional, Set, Tuple
from neo4j import GraphDatabase
import os
from src.infrastructure.logging import RAGLogger

logger = RAGLogger.get_logger('neo4j_ops')


class Neo4jRepository:
    """
    Repository for managing graph data in Neo4j.
    
    Graph Schema:
    - Nodes:
      * Document (id, file_name, source_type)
      * Chunk (id, document_id, chunk_index, content_preview)
      * Entity (id, name, type, properties)
    - Relationships:
      * (Document)-[:CONTAINS]->(Chunk)
      * (Chunk)-[:MENTIONS]->(Entity)
      * (Entity)-[:RELATED_TO]->(Entity)
      * (Chunk)-[:CO_OCCURS_WITH]->(Chunk)
      * (Chunk)-[:SHARES_ENTITY]->(Chunk) - chunks that mention the same entities
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize Neo4j repository.
        
        Args:
            uri: Neo4j connection URI (defaults to NEO4J_URI env var or localhost)
            user: Neo4j username (defaults to NEO4J_USER env var or 'neo4j')
            password: Neo4j password (defaults to NEO4J_PASSWORD env var or 'password')
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        # Verify connection and create constraints
        self._initialize_database()
        
        logger.info(f"Neo4j repository initialized: uri={self.uri}, user={self.user}")

    def _initialize_database(self):
        """Create indexes and constraints for better performance."""
        with self.driver.session() as session:
            # Create constraints for uniqueness
            constraints = [
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist, which is fine
                    logger.debug(f"Constraint creation (may already exist): {str(e)}")
            
            # Create indexes for better query performance
            indexes = [
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX chunk_document_id IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index creation (may already exist): {str(e)}")
            
            logger.info("Neo4j database initialized with constraints and indexes")

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")

    def add_document(self, document_id: str, file_name: Optional[str], source_type: str):
        """
        Add a document node to the graph.
        
        Args:
            document_id: Unique document identifier
            file_name: Name of the file
            source_type: Type of source (pdf, text, clipboard)
        """
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $document_id})
                SET d.file_name = $file_name,
                    d.source_type = $source_type,
                    d.created_at = datetime()
                """,
                document_id=document_id,
                file_name=file_name or "",
                source_type=source_type,
            )
            logger.debug(f"Document node added/updated: document_id={document_id}")

    def add_chunk(
        self,
        chunk_id: str,
        document_id: str,
        chunk_index: int,
        content_preview: str,
    ):
        """
        Add a chunk node and link it to its document.
        
        Args:
            chunk_id: Unique chunk identifier
            document_id: ID of the parent document
            chunk_index: Index of the chunk in the document
            content_preview: Preview of chunk content (first 200 chars)
        """
        with self.driver.session() as session:
            # Create chunk node
            session.run(
                """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.document_id = $document_id,
                    c.chunk_index = $chunk_index,
                    c.content_preview = $content_preview
                """,
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                content_preview=content_preview[:200],
            )
            
            # Link chunk to document
            session.run(
                """
                MATCH (d:Document {id: $document_id})
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (d)-[:CONTAINS]->(c)
                """,
                document_id=document_id,
                chunk_id=chunk_id,
            )
            logger.debug(f"Chunk node added: chunk_id={chunk_id}, document_id={document_id}")

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: Optional[Dict] = None,
    ):
        """
        Add an entity node to the graph.
        
        Args:
            entity_id: Unique entity identifier
            name: Name of the entity
            entity_type: Type of entity (Person, Organization, Concept, etc.)
            properties: Additional properties as dictionary (will be stored as JSON string)
        """
        import json
        with self.driver.session() as session:
            # Convert properties to JSON string since Neo4j doesn't support nested objects
            props_json = json.dumps(properties) if properties else "{}"
            session.run(
                """
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name,
                    e.type = $entity_type,
                    e.properties_json = $properties_json
                """,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                properties_json=props_json,
            )
            logger.debug(f"Entity node added: entity_id={entity_id}, name={name}, type={entity_type}")

    def link_chunk_to_entity(self, chunk_id: str, entity_id: str, confidence: float = 1.0):
        """
        Create a MENTIONS relationship between a chunk and an entity.
        
        Args:
            chunk_id: ID of the chunk
            entity_id: ID of the entity
            confidence: Confidence score for the relationship
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (c)-[r:MENTIONS]->(e)
                SET r.confidence = $confidence,
                    r.created_at = datetime()
                """,
                chunk_id=chunk_id,
                entity_id=entity_id,
                confidence=confidence,
            )
            logger.debug(f"Linked chunk to entity: chunk_id={chunk_id}, entity_id={entity_id}")

    def link_entities(
        self,
        entity1_id: str,
        entity2_id: str,
        relationship_type: str = "RELATED_TO",
        properties: Optional[Dict] = None,
    ):
        """
        Create a relationship between two entities.
        
        Args:
            entity1_id: ID of the first entity
            entity2_id: ID of the second entity
            relationship_type: Type of relationship
            properties: Additional properties for the relationship (will be stored as JSON string)
        """
        import json
        with self.driver.session() as session:
            # Convert properties to JSON string since Neo4j doesn't support nested objects
            props_json = json.dumps(properties) if properties else "{}"
            session.run(
                f"""
                MATCH (e1:Entity {{id: $entity1_id}})
                MATCH (e2:Entity {{id: $entity2_id}})
                MERGE (e1)-[r:{relationship_type}]->(e2)
                SET r.properties_json = $properties_json,
                    r.created_at = datetime()
                """,
                entity1_id=entity1_id,
                entity2_id=entity2_id,
                properties_json=props_json,
            )
            logger.debug(
                f"Linked entities: entity1_id={entity1_id}, entity2_id={entity2_id}, "
                f"relationship={relationship_type}"
            )

    def link_co_occurring_chunks(self, chunk1_id: str, chunk2_id: str, weight: float = 1.0):
        """
        Create a CO_OCCURS_WITH relationship between two chunks.
        
        Args:
            chunk1_id: ID of the first chunk
            chunk2_id: ID of the second chunk
            weight: Weight of the co-occurrence
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (c1:Chunk {id: $chunk1_id})
                MATCH (c2:Chunk {id: $chunk2_id})
                WHERE c1.id <> c2.id
                MERGE (c1)-[r:CO_OCCURS_WITH]->(c2)
                SET r.weight = $weight,
                    r.created_at = datetime()
                """,
                chunk1_id=chunk1_id,
                chunk2_id=chunk2_id,
                weight=weight,
            )

    def link_chunks_sharing_entities(
        self,
        chunk_ids: List[str],
        min_shared_entities: int = 1,
        link_across_documents: bool = True,
    ):
        """
        Create relationships between chunks that mention the same entities.
        
        This creates SHARES_ENTITY relationships between chunks that mention
        common entities, with a weight based on the number of shared entities.
        
        Args:
            chunk_ids: List of chunk IDs to process
            min_shared_entities: Minimum number of shared entities to create a relationship
            link_across_documents: If True, also link to chunks from other documents
        """
        if len(chunk_ids) < 2:
            logger.debug(f"Skipping chunk linking - only {len(chunk_ids)} chunk(s)")
            return
        
        try:
            with self.driver.session() as session:
                # First, check if chunks have entities
                check_result = session.run(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE c.id IN $chunk_ids
                    RETURN count(DISTINCT c) as chunks_with_entities, count(DISTINCT e) as total_entities
                    """,
                    chunk_ids=chunk_ids,
                )
                check_record = check_result.single()
                if check_record:
                    chunks_with_entities = check_record["chunks_with_entities"]
                    total_entities = check_record["total_entities"]
                    logger.info(
                        f"Chunk linking check: {chunks_with_entities} chunks have entities "
                        f"({total_entities} total entities)"
                    )
                
                # Find pairs of chunks that mention the same entities
                if link_across_documents:
                    # Link chunks from any document
                    query = """
                    MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
                    WHERE c1.id IN $chunk_ids
                      AND c1.id <> c2.id
                    WITH c1, c2, count(DISTINCT e) as shared_entity_count
                    WHERE shared_entity_count >= $min_shared_entities
                    MERGE (c1)-[r:SHARES_ENTITY]->(c2)
                    SET r.shared_entity_count = shared_entity_count,
                        r.weight = shared_entity_count,
                        r.created_at = datetime()
                    RETURN c1.id as chunk1_id, c2.id as chunk2_id, shared_entity_count, 
                           c1.document_id as doc1_id, c2.document_id as doc2_id
                    """
                else:
                    # Only link chunks from the same document
                    query = """
                    MATCH (c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)
                    WHERE c1.id IN $chunk_ids
                      AND c2.id IN $chunk_ids
                      AND c1.id <> c2.id
                    WITH c1, c2, count(DISTINCT e) as shared_entity_count
                    WHERE shared_entity_count >= $min_shared_entities
                    MERGE (c1)-[r:SHARES_ENTITY]->(c2)
                    SET r.shared_entity_count = shared_entity_count,
                        r.weight = shared_entity_count,
                        r.created_at = datetime()
                    RETURN c1.id as chunk1_id, c2.id as chunk2_id, shared_entity_count,
                           c1.document_id as doc1_id, c2.document_id as doc2_id
                    """
                
                result = session.run(
                    query,
                    chunk_ids=chunk_ids,
                    min_shared_entities=min_shared_entities,
                )
                
                count = 0
                same_doc_count = 0
                cross_doc_count = 0
                for record in result:
                    count += 1
                    doc1_id = record.get("doc1_id")
                    doc2_id = record.get("doc2_id")
                    is_same_doc = doc1_id == doc2_id
                    
                    if is_same_doc:
                        same_doc_count += 1
                    else:
                        cross_doc_count += 1
                    
                    logger.info(
                        f"Linked chunks sharing entities: {record['chunk1_id']} <-> "
                        f"{record['chunk2_id']} ({record['shared_entity_count']} shared entities) "
                        f"[{'same doc' if is_same_doc else 'cross-doc'}]"
                    )
                
                if count > 0:
                    logger.info(
                        f"Created {count} SHARES_ENTITY relationships: "
                        f"{same_doc_count} same-document, {cross_doc_count} cross-document "
                        f"(min shared entities: {min_shared_entities})"
                    )
                else:
                    logger.warning(
                        f"No SHARES_ENTITY relationships created for {len(chunk_ids)} chunks. "
                        f"This might mean chunks don't share entities or entities weren't extracted."
                    )
                    
        except Exception as e:
            logger.error(f"Error linking chunks sharing entities: {str(e)}", exc_info=True)
            raise

    def find_related_chunks(
        self,
        chunk_ids: List[str],
        max_depth: int = 2,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Find chunks related to the given chunks through graph traversal.
        
        This finds chunks that:
        1. Mention the same entities as the input chunks
        2. Are connected through entity relationships
        3. Co-occur with the input chunks
        
        Args:
            chunk_ids: List of chunk IDs to start from
            max_depth: Maximum depth for graph traversal
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with chunk information and relevance score
        """
        if not chunk_ids:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (startChunk:Chunk)
                WHERE startChunk.id IN $chunk_ids
                
                // Find chunks that mention the same entities (via SHARES_ENTITY relationship)
                OPTIONAL MATCH (startChunk)-[:SHARES_ENTITY]-(relatedChunk:Chunk)
                WHERE NOT relatedChunk.id IN $chunk_ids
                
                // Also find chunks that mention the same entities (direct entity matching)
                OPTIONAL MATCH (startChunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(relatedChunk1:Chunk)
                WHERE NOT relatedChunk1.id IN $chunk_ids
                
                // Find chunks connected through entity relationships
                OPTIONAL MATCH (startChunk)-[:MENTIONS]->(e1:Entity)-[:RELATED_TO*1..2]-(e2:Entity)<-[:MENTIONS]-(relatedChunk2:Chunk)
                WHERE NOT relatedChunk2.id IN $chunk_ids
                
                // Find co-occurring chunks
                OPTIONAL MATCH (startChunk)-[:CO_OCCURS_WITH]-(relatedChunk3:Chunk)
                WHERE NOT relatedChunk3.id IN $chunk_ids
                
                // Combine all related chunks and collect source chunks
                WITH DISTINCT 
                    COALESCE(relatedChunk, relatedChunk1, relatedChunk2, relatedChunk3) AS chunk,
                    collect(DISTINCT startChunk.id) as source_chunk_ids
                WHERE chunk IS NOT NULL
                
                // Calculate relevance score based on number of connections
                // Check SHARES_ENTITY relationships (weighted by shared entity count)
                OPTIONAL MATCH (chunk)-[r:SHARES_ENTITY]-(startChunk:Chunk)
                WHERE startChunk.id IN $chunk_ids
                WITH chunk, source_chunk_ids, sum(COALESCE(r.shared_entity_count, 0)) as shares_entity_score
                
                // Check direct entity connections
                OPTIONAL MATCH (chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(startChunk:Chunk)
                WHERE startChunk.id IN $chunk_ids
                WITH chunk, source_chunk_ids, shares_entity_score, count(DISTINCT e) as entity_connections
                
                // Check co-occurrence
                OPTIONAL MATCH (chunk)-[:CO_OCCURS_WITH]-(startChunk:Chunk)
                WHERE startChunk.id IN $chunk_ids
                WITH chunk, source_chunk_ids, shares_entity_score, entity_connections, count(DISTINCT startChunk) as co_occur_count
                
                RETURN chunk.id as chunk_id,
                       chunk.document_id as document_id,
                       chunk.chunk_index as chunk_index,
                       chunk.content_preview as content_preview,
                       (shares_entity_score * 3 + entity_connections * 2 + co_occur_count) as relevance_score,
                       source_chunk_ids
                ORDER BY relevance_score DESC
                LIMIT $limit
                """,
                chunk_ids=chunk_ids,
                limit=limit,
            )
            
            results = []
            for record in result:
                results.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "content_preview": record["content_preview"],
                    "relevance_score": record["relevance_score"],
                    "source_chunk_ids": record.get("source_chunk_ids", []),
                })
            
            logger.info(f"Found {len(results)} related chunks for {len(chunk_ids)} input chunks")
            return results

    def find_entities_by_query(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """
        Find entities that match the query text.
        
        Args:
            query: Query text to search for
            entity_types: Optional list of entity types to filter by
            limit: Maximum number of results
            
        Returns:
            List of entity dictionaries
        """
        with self.driver.session() as session:
            if entity_types:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($search_text)
                      AND e.type IN $entity_types
                    RETURN e.id as entity_id,
                           e.name as name,
                           e.type as type,
                           e.properties_json as properties_json
                    LIMIT $limit
                    """,
                    search_text=query,
                    entity_types=entity_types,
                    limit=limit,
                )
            else:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($search_text)
                    RETURN e.id as entity_id,
                           e.name as name,
                           e.type as type,
                           e.properties_json as properties_json
                    LIMIT $limit
                    """,
                    search_text=query,
                    limit=limit,
                )
            
            import json
            results = []
            for record in result:
                props_json = record.get("properties_json") or "{}"
                try:
                    properties = json.loads(props_json) if isinstance(props_json, str) else (props_json or {})
                except (json.JSONDecodeError, TypeError):
                    properties = {}
                results.append({
                    "entity_id": record["entity_id"],
                    "name": record["name"],
                    "type": record["type"],
                    "properties": properties,
                })
            
            return results

    def find_entity_by_name(
        self,
        name: str,
        entity_type: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Find an entity by exact name match (case-insensitive).
        
        This is used for entity resolution - finding existing entities
        that might have been created in other chunks.
        
        Args:
            name: Entity name to search for
            entity_type: Optional entity type to filter by
            
        Returns:
            Entity dictionary if found, None otherwise
        """
        with self.driver.session() as session:
            if entity_type:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) = toLower($name)
                      AND e.type = $entity_type
                    RETURN e.id as entity_id,
                           e.name as name,
                           e.type as type,
                           e.properties_json as properties_json
                    LIMIT 1
                    """,
                    name=name,
                    entity_type=entity_type,
                )
            else:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) = toLower($name)
                    RETURN e.id as entity_id,
                           e.name as name,
                           e.type as type,
                           e.properties_json as properties_json
                    LIMIT 1
                    """,
                    name=name,
                )
            
            import json
            record = result.single()
            if record:
                props_json = record.get("properties_json") or "{}"
                try:
                    properties = json.loads(props_json) if isinstance(props_json, str) else (props_json or {})
                except (json.JSONDecodeError, TypeError):
                    properties = {}
                return {
                    "entity_id": record["entity_id"],
                    "name": record["name"],
                    "type": record["type"],
                    "properties": properties,
                }
            return None

    def get_chunks_by_entities(
        self,
        entity_ids: List[str],
        limit: int = 10,
    ) -> List[Dict]:
        """
        Find chunks that mention the given entities.
        
        Args:
            entity_ids: List of entity IDs
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunk dictionaries
        """
        if not entity_ids:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
                WHERE e.id IN $entity_ids
                RETURN DISTINCT c.id as chunk_id,
                       c.document_id as document_id,
                       c.chunk_index as chunk_index,
                       c.content_preview as content_preview,
                       count(DISTINCT e) as entity_count
                ORDER BY entity_count DESC
                LIMIT $limit
                """,
                entity_ids=entity_ids,
                limit=limit,
            )
            
            results = []
            for record in result:
                results.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "content_preview": record["content_preview"],
                    "entity_count": record["entity_count"],
                })
            
            return results

    def get_entities_for_chunks(
        self,
        chunk_ids: List[str],
    ) -> Dict[str, List[Dict]]:
        """
        Get all entities mentioned in the given chunks.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            Dictionary mapping chunk_id to list of entity dictionaries
        """
        if not chunk_ids:
            return {}
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.id IN $chunk_ids
                RETURN c.id as chunk_id,
                       e.id as entity_id,
                       e.name as entity_name,
                       e.type as entity_type,
                       e.properties as entity_properties
                ORDER BY c.id, e.name
                """,
                chunk_ids=chunk_ids,
            )
            
            import json
            chunk_entities = {}
            for record in result:
                chunk_id = record["chunk_id"]
                if chunk_id not in chunk_entities:
                    chunk_entities[chunk_id] = []
                
                props_json = record.get("entity_properties_json") or "{}"
                try:
                    properties = json.loads(props_json) if isinstance(props_json, str) else (props_json or {})
                except (json.JSONDecodeError, TypeError):
                    properties = {}
                
                chunk_entities[chunk_id].append({
                    "entity_id": record["entity_id"],
                    "name": record["entity_name"],
                    "type": record["entity_type"],
                    "properties": properties,
                })
            
            return chunk_entities

    def get_related_entities(
        self,
        entity_ids: List[str],
        max_depth: int = 2,
        limit: int = 20,
    ) -> List[Dict]:
        """
        Get entities related to the given entities through relationships.
        
        Args:
            entity_ids: List of entity IDs
            max_depth: Maximum depth for relationship traversal
            limit: Maximum number of related entities to return
            
        Returns:
            List of related entity dictionaries with relationship info
        """
        if not entity_ids:
            return []
        
        with self.driver.session() as session:
            # Neo4j doesn't allow parameters in variable-length patterns, so we need to build the query
            # Limit max_depth to a reasonable value for safety
            max_depth = min(max_depth, 5)  # Cap at 5 for safety
            
            # Build the relationship pattern based on max_depth
            if max_depth == 1:
                rel_pattern = "RELATED_TO"
            else:
                rel_pattern = f"RELATED_TO*1..{max_depth}"
            
            query = f"""
                MATCH (startEntity:Entity)
                WHERE startEntity.id IN $entity_ids
                
                // Find directly related entities using variable-length pattern
                MATCH path = (startEntity)-[:{rel_pattern}]-(relatedEntity:Entity)
                WHERE NOT relatedEntity.id IN $entity_ids
                
                WITH DISTINCT relatedEntity, 
                     startEntity,
                     length(path) as path_length
                
                RETURN DISTINCT relatedEntity.id as entity_id,
                       relatedEntity.name as entity_name,
                       relatedEntity.type as entity_type,
                       relatedEntity.properties_json as entity_properties_json,
                       min(path_length) as min_path_length,
                       count(DISTINCT startEntity) as connection_count
                ORDER BY connection_count DESC, min_path_length ASC
                LIMIT $limit
                """
            
            result = session.run(
                query,
                parameters={
                    "entity_ids": entity_ids,
                    "limit": limit,
                }
            )
            
            import json
            results = []
            for record in result:
                props_json = record.get("entity_properties_json") or "{}"
                try:
                    properties = json.loads(props_json) if isinstance(props_json, str) else (props_json or {})
                except (json.JSONDecodeError, TypeError):
                    properties = {}
                results.append({
                    "entity_id": record["entity_id"],
                    "name": record["entity_name"],
                    "type": record["entity_type"],
                    "properties": properties,
                    "path_length": record["min_path_length"],
                    "connection_count": record["connection_count"],
                })
            
            return results

    def get_entity_relationships(
        self,
        entity_ids: List[str],
        limit: int = 50,
    ) -> List[Dict]:
        """
        Get relationships between entities in the given list.
        
        Args:
            entity_ids: List of entity IDs
            limit: Maximum number of relationships to return
            
        Returns:
            List of relationship dictionaries
        """
        if not entity_ids:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
                RETURN e1.id as source_entity_id,
                       e1.name as source_entity_name,
                       e1.type as source_entity_type,
                       type(r) as relationship_type,
                       r.properties_json as relationship_properties_json,
                       e2.id as target_entity_id,
                       e2.name as target_entity_name,
                       e2.type as target_entity_type
                LIMIT $limit
                """,
                entity_ids=entity_ids,
                limit=limit,
            )
            
            import json
            results = []
            for record in result:
                props_json = record.get("relationship_properties_json") or "{}"
                try:
                    properties = json.loads(props_json) if isinstance(props_json, str) else (props_json or {})
                except (json.JSONDecodeError, TypeError):
                    properties = {}
                results.append({
                    "source_entity_id": record["source_entity_id"],
                    "source_entity_name": record["source_entity_name"],
                    "source_entity_type": record["source_entity_type"],
                    "relationship_type": record["relationship_type"],
                    "relationship_properties": properties,
                    "target_entity_id": record["target_entity_id"],
                    "target_entity_name": record["target_entity_name"],
                    "target_entity_type": record["target_entity_type"],
                })
            
            return results

    def delete_document(self, document_id: str):
        """
        Delete a document and all its associated nodes and relationships.
        
        Args:
            document_id: ID of the document to delete
        """
        with self.driver.session() as session:
            # Delete document and all related nodes
            session.run(
                """
                MATCH (d:Document {id: $document_id})
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                OPTIONAL MATCH (c)-[r1]->()
                OPTIONAL MATCH ()-[r2]->(c)
                DELETE d, c, r1, r2
                """,
                document_id=document_id,
            )
            logger.info(f"Deleted document from graph: document_id={document_id}")

    def get_graph_stats(self) -> Dict:
        """
        Get statistics about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)
                WITH count(d) as doc_count
                MATCH (c:Chunk)
                WITH doc_count, count(c) as chunk_count
                MATCH (e:Entity)
                WITH doc_count, chunk_count, count(e) as entity_count
                MATCH ()-[r]->()
                RETURN doc_count, chunk_count, entity_count, count(r) as relationship_count
                """
            )
            
            record = result.single()
            if record:
                return {
                    "documents": record["doc_count"],
                    "chunks": record["chunk_count"],
                    "entities": record["entity_count"],
                    "relationships": record["relationship_count"],
                }
            return {
                "documents": 0,
                "chunks": 0,
                "entities": 0,
                "relationships": 0,
            }
