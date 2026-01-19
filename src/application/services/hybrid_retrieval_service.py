"""
Hybrid Retrieval Service - Combines vector search with graph traversal.

This service implements the hybrid Graph RAG approach:
1. First, find similar chunks using ChromaDB vector search
2. Then, enrich context by finding related chunks through Neo4j graph
"""
from typing import List, Tuple, Dict, Set
from src.domain.entities.document import Document
from src.domain.repositories.document_repository import IDocumentRepository
from src.infrastructure.graph_store.neo4j_repository import Neo4jRepository
from src.application.services.entity_extraction_service import EntityExtractionService
from src.infrastructure.logging import RAGLogger

logger = RAGLogger.get_logger('hybrid_retrieval')


class HybridRetrievalService:
    """
    Service that combines vector search (ChromaDB) with graph traversal (Neo4j).
    
    Workflow:
    1. Vector search: Find similar chunks using ChromaDB
    2. Graph enrichment: Find related chunks through Neo4j graph
    3. Combine and deduplicate results
    """

    def __init__(
        self,
        document_repository: IDocumentRepository,
        neo4j_repository: Neo4jRepository,
        entity_extraction_service: EntityExtractionService,
        vector_top_k: int = 5,
        graph_top_k: int = 5,
        similarity_threshold: float = 0.4,
    ):
        """
        Initialize the hybrid retrieval service.
        
        Args:
            document_repository: ChromaDB document repository
            neo4j_repository: Neo4j graph repository
            entity_extraction_service: Service for entity extraction
            vector_top_k: Number of top results from vector search
            graph_top_k: Number of additional results from graph traversal
            similarity_threshold: Minimum similarity score for vector results
        """
        self.document_repository = document_repository
        self.neo4j_repository = neo4j_repository
        self.entity_extraction_service = entity_extraction_service
        self.vector_top_k = vector_top_k
        self.graph_top_k = graph_top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
    ) -> Tuple[List[Tuple[Document, float]], List[Dict], Dict]:
        """
        Perform hybrid retrieval combining vector search and graph traversal.
        
        Args:
            query: User query
            
        Returns:
            Tuple of:
            - List of (document, score) tuples from hybrid retrieval
            - List of source details including graph information
            - Graph context dictionary with entities and relationships
        """
        logger.info(f"Starting hybrid retrieval: query_length={len(query)}")
        
        # Step 1: Vector search using ChromaDB
        vector_results = self.document_repository.search_similar(
            query=query,
            top_k=self.vector_top_k,
        )
        
        # Filter by similarity threshold and mark as vector sources
        relevant_vector_results = []
        for doc, score in vector_results:
            if score >= self.similarity_threshold:
                # Ensure vector chunks are explicitly marked
                if "source_type" not in doc.metadata:
                    doc.metadata["source_type"] = "vector"
                doc.metadata["from_graph"] = False
                relevant_vector_results.append((doc, score))
        
        logger.info(
            f"Vector search found {len(relevant_vector_results)} relevant chunks "
            f"(from {len(vector_results)} total)"
        )
        
        if not relevant_vector_results:
            logger.warning("No relevant chunks found from vector search")
            return [], [], {}
        
        # Step 2: Extract entities from query and retrieved chunks
        query_entities = self.entity_extraction_service.extract_entities_from_query(query)
        logger.info(f"Extracted {len(query_entities)} entities from query")
        
        # Step 3: Find related chunks and entities through graph traversal
        graph_results = []
        graph_entities = []
        graph_relationships = []
        
        # Get chunk IDs from vector results
        vector_chunk_ids = [
            f"{doc.id}_{doc.metadata.get('chunk_index', 0)}"
            for doc, _ in relevant_vector_results
        ]
        
        # Get entities mentioned in the vector chunks
        chunk_entities_map = self.neo4j_repository.get_entities_for_chunks(vector_chunk_ids)
        all_entity_ids = set()
        for entities in chunk_entities_map.values():
            all_entity_ids.update([e["entity_id"] for e in entities])
        
        logger.info(f"Found {len(all_entity_ids)} entities in vector chunks")
        
        # Get related entities through relationships
        if all_entity_ids:
            related_entities = self.neo4j_repository.get_related_entities(
                entity_ids=list(all_entity_ids),
                max_depth=2,
                limit=20,
            )
            graph_entities = related_entities
            logger.info(f"Found {len(related_entities)} related entities through graph")
            
            # Get relationships between entities
            all_related_entity_ids = list(all_entity_ids) + [e["entity_id"] for e in related_entities]
            relationships = self.neo4j_repository.get_entity_relationships(
                entity_ids=all_related_entity_ids,
                limit=50,
            )
            graph_relationships = relationships
            logger.info(f"Found {len(relationships)} entity relationships")
        
        # Find related chunks via graph (structurally connected)
        related_chunks = self.neo4j_repository.find_related_chunks(
            chunk_ids=vector_chunk_ids,
            max_depth=2,
            limit=self.graph_top_k,
        )
        
        logger.info(f"Graph traversal found {len(related_chunks)} structurally related chunks")
        
        # Step 4: Also search by entities from query
        entity_based_chunks = []
        if query_entities:
            # Find entities in graph
            entity_ids = []
            for entity in query_entities:
                # Try to find matching entities in graph
                found_entities = self.neo4j_repository.find_entities_by_query(
                    query=entity["name"],
                    entity_types=[entity.get("type", "Other")],
                    limit=5,
                )
                entity_ids.extend([e["entity_id"] for e in found_entities])
            
            if entity_ids:
                # Get chunks that mention these entities
                entity_chunks = self.neo4j_repository.get_chunks_by_entities(
                    entity_ids=entity_ids,
                    limit=self.graph_top_k,
                )
                entity_based_chunks = entity_chunks
                logger.info(f"Found {len(entity_based_chunks)} chunks via entity search")
        
        # Step 5: Combine and deduplicate results
        # Create a mapping from chunk_id to vector score for quick lookup
        vector_chunk_score_map = {}
        for doc, score in relevant_vector_results:
            chunk_id = f"{doc.id}_{doc.metadata.get('chunk_index', 0)}"
            vector_chunk_score_map[chunk_id] = score
        
        # First, mark all vector results with source_type="vector"
        all_chunk_ids = set(vector_chunk_ids)
        combined_results = []
        
        # Add vector results first (these have priority)
        for doc, score in relevant_vector_results:
            # Ensure vector chunks are marked correctly
            if "source_type" not in doc.metadata:
                doc.metadata["source_type"] = "vector"
            combined_results.append((doc, score))
        
        # Add graph results (avoid duplicates, these are secondary/enrichment)
        for graph_chunk in related_chunks + entity_based_chunks:
            chunk_id = graph_chunk["chunk_id"]
            if chunk_id not in all_chunk_ids:
                all_chunk_ids.add(chunk_id)
                
                # Get the full chunk content from ChromaDB
                doc_id = graph_chunk["document_id"]
                chunk_index = graph_chunk.get("chunk_index", 0)
                
                # Try to get the chunk from ChromaDB
                chunk_content = graph_chunk.get("content_preview", "")
                if hasattr(self.document_repository, 'collection'):
                    # Access ChromaDB collection directly to get chunk
                    try:
                        chunk_result = self.document_repository.collection.get(
                            ids=[chunk_id]
                        )
                        if chunk_result["documents"] and chunk_result["documents"][0]:
                            chunk_content = chunk_result["documents"][0]
                    except Exception as e:
                        logger.debug(f"Could not get chunk content from ChromaDB: {str(e)}")
                
                # Get document metadata
                full_doc = self.document_repository.get_by_id(doc_id)
                if full_doc:
                    partial_doc = Document(
                        id=doc_id,
                        content=chunk_content,
                        source_type=full_doc.source_type,
                        file_name=full_doc.file_name,
                        metadata={
                            "chunk_index": chunk_index,
                            "from_graph": True,
                            "source_type": "graph",  # Explicitly mark as graph source
                        },
                    )
                    
                    # Calculate graph relevance relative to source vector chunks
                    # Find the highest relevance score among source vector chunks
                    source_chunk_ids = graph_chunk.get("source_chunk_ids", [])
                    max_source_score = 0.0
                    
                    if source_chunk_ids:
                        # Find the highest score among source vector chunks
                        for source_id in source_chunk_ids:
                            if source_id in vector_chunk_score_map:
                                max_source_score = max(max_source_score, vector_chunk_score_map[source_id])
                    
                    # If no source chunks found (e.g., entity-based chunks), use average vector score as fallback
                    if max_source_score == 0.0 and relevant_vector_results:
                        max_source_score = sum(score for _, score in relevant_vector_results) / len(relevant_vector_results)
                    
                    # Normalize graph relevance score
                    # For chunks from find_related_chunks: use relevance_score
                    # For chunks from get_chunks_by_entities: use entity_count as proxy
                    if "relevance_score" in graph_chunk:
                        # Normalize graph relevance score (0-1 range, typically 0.1-0.5)
                        graph_relevance = min(graph_chunk.get("relevance_score", 1) / 10.0, 0.5)
                    elif "entity_count" in graph_chunk:
                        # For entity-based chunks, use entity_count as relevance proxy
                        entity_count = graph_chunk.get("entity_count", 1)
                        graph_relevance = min(entity_count / 10.0, 0.5)
                    else:
                        # Default fallback
                        graph_relevance = 0.3
                    
                    # Multiply graph relevance by source vector chunk's relevance
                    # This ensures graph chunks always have lower relevance than their source vector chunks
                    final_graph_score = graph_relevance * max_source_score
                    
                    # Ensure it's always lower than the source score (at most 90% of source)
                    if max_source_score > 0:
                        final_graph_score = min(final_graph_score, max_source_score * 0.9)
                    
                    logger.debug(
                        f"Graph chunk {chunk_id}: graph_relevance={graph_relevance:.3f}, "
                        f"source_score={max_source_score:.3f}, final_score={final_graph_score:.3f}"
                    )
                    
                    combined_results.append((partial_doc, final_graph_score))
        
        # Sort by score (descending)
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Build source details
        sources = []
        for doc, score in combined_results:
            chunk_index = doc.metadata.get("chunk_index", 0)
            is_from_graph = doc.metadata.get("from_graph", False)
            
            sources.append({
                "document_id": doc.id,
                "chunk_index": chunk_index,
                "relevance_score": score,
                "file_name": doc.file_name,
                "source_type": "graph" if is_from_graph else "vector",
            })
        
        logger.info(
            f"Hybrid retrieval completed: {len(combined_results)} total chunks "
            f"({len(relevant_vector_results)} vector, "
            f"{len(combined_results) - len(relevant_vector_results)} graph), "
            f"{len(graph_entities)} related entities, "
            f"{len(graph_relationships)} relationships"
        )
        
        # Store graph context for context building
        graph_context = {
            "entities": graph_entities,
            "relationships": graph_relationships,
            "chunk_entities": chunk_entities_map,
        }
        
        return combined_results, sources, graph_context
