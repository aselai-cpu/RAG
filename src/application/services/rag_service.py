"""
RAG Service - Application layer orchestrating the RAG workflow.

This service coordinates between the domain, infrastructure, and
implements the core RAG (Retrieval-Augmented Generation) logic.
Now enhanced with Graph RAG capabilities using Neo4j.
"""
from typing import List, Optional, Tuple, Dict
from src.domain.entities.document import Document
from src.domain.entities.chat import Message, ChatSession
from src.domain.repositories.document_repository import IDocumentRepository
from src.infrastructure.llm.openai_service import OpenAIService
from src.application.services.hybrid_retrieval_service import HybridRetrievalService
from src.infrastructure.logging import RAGLogger

logger = RAGLogger.get_logger('rag_service')


class RAGService:
    """
    Service orchestrating the Retrieval-Augmented Generation workflow.

    This implements the classic RAG pattern:
    1. User asks a question
    2. Retrieve relevant documents from vector store
    3. Augment the prompt with retrieved context
    4. Generate response using LLM
    """

    def __init__(
        self,
        document_repository: IDocumentRepository,
        llm_service: OpenAIService,
        top_k_retrieval: int = 5,
        similarity_threshold: float = 0.5,
        hybrid_retrieval_service: Optional[HybridRetrievalService] = None,
    ):
        """
        Initialize the RAG service.

        Args:
            document_repository: Repository for document storage and retrieval
            llm_service: Service for LLM interactions
            top_k_retrieval: Number of top documents to retrieve
            similarity_threshold: Minimum similarity score for retrieval
            hybrid_retrieval_service: Optional hybrid retrieval service for Graph RAG
        """
        self.document_repository = document_repository
        self.llm_service = llm_service
        self.top_k_retrieval = top_k_retrieval
        self.similarity_threshold = similarity_threshold
        self.hybrid_retrieval_service = hybrid_retrieval_service
        self.use_graph_rag = hybrid_retrieval_service is not None

    def query(
        self,
        question: str,
        chat_history: Optional[List[Message]] = None,
    ) -> Tuple[str, List[str], List[dict]]:
        """
        Process a RAG query.

        This implements the core RAG workflow:
        1. Retrieve relevant documents
        2. Build context from retrieved documents
        3. Generate response with context
        4. Return response, source document IDs, and detailed source information

        Args:
            question: The user's question
            chat_history: Optional chat history for context

        Returns:
            Tuple of (response_text, list_of_source_document_ids, list_of_source_details)
            Source details include: document_id, chunk_index, relevance_score, file_name
        """
        # Step 1: Retrieve relevant documents (using hybrid retrieval if available)
        graph_context = None
        if self.use_graph_rag and self.hybrid_retrieval_service:
            logger.info("Using Graph RAG hybrid retrieval")
            relevant_docs, sources, graph_context = self.hybrid_retrieval_service.retrieve(question)
        else:
            logger.info("Using standard vector retrieval")
            retrieved_docs = self.document_repository.search_similar(
                query=question, top_k=self.top_k_retrieval
            )

            # Filter by similarity threshold
            relevant_docs = [
                (doc, score)
                for doc, score in retrieved_docs
                if score >= self.similarity_threshold
            ]
            
            # Build sources list
            sources = []
            for doc, score in relevant_docs:
                chunk_index = doc.metadata.get("chunk_index", 0)
                sources.append({
                    "document_id": doc.id,
                    "chunk_index": chunk_index,
                    "relevance_score": score,
                    "file_name": doc.file_name,
                    "source_type": "vector",
                })

        # Step 2: Build context from retrieved documents and graph information
        context = self._build_context(relevant_docs, graph_context=graph_context)

        # Step 3: Build messages for LLM
        messages = []

        # Add recent chat history if provided (last 5 messages)
        if chat_history:
            recent_history = chat_history[-5:]
            for msg in recent_history:
                messages.append({"role": msg.role, "content": msg.content})

        # Add current question
        messages.append({"role": "user", "content": question})

        # Step 4: Generate response
        response = self.llm_service.generate_response(
            messages=messages, context=context if relevant_docs else None
        )

        # Sources are already built above (either from hybrid or standard retrieval)
        # Also return unique document IDs for backward compatibility
        source_ids = list(set(doc.id for doc, _ in relevant_docs))

        return response, source_ids, sources

    def query_stream(
        self,
        question: str,
        chat_history: Optional[List[Message]] = None,
    ):
        """
        Process a RAG query with streaming response.

        Similar to query() but yields response chunks as they're generated.

        Args:
            question: The user's question
            chat_history: Optional chat history for context

        Yields:
            Tuples of (response_chunk, source_ids, source_details)
            The source_ids and source_details will be sent with the first chunk
            Source details include: document_id, chunk_index, relevance_score, file_name
        """
        # Retrieve relevant documents (using hybrid retrieval if available)
        graph_context = None
        if self.use_graph_rag and self.hybrid_retrieval_service:
            logger.info("Using Graph RAG hybrid retrieval (streaming)")
            relevant_docs, sources, graph_context = self.hybrid_retrieval_service.retrieve(question)
        else:
            logger.info("Using standard vector retrieval (streaming)")
            retrieved_docs = self.document_repository.search_similar(
                query=question, top_k=self.top_k_retrieval
            )

            # Filter by similarity threshold
            relevant_docs = [
                (doc, score)
                for doc, score in retrieved_docs
                if score >= self.similarity_threshold
            ]
            
            # Build sources list
            sources = []
            for doc, score in relevant_docs:
                chunk_index = doc.metadata.get("chunk_index", 0)
                sources.append({
                    "document_id": doc.id,
                    "chunk_index": chunk_index,
                    "relevance_score": score,
                    "file_name": doc.file_name,
                    "source_type": "vector",
                })

        # Build context
        context = self._build_context(relevant_docs, graph_context=graph_context)

        # Build messages
        messages = []
        if chat_history:
            recent_history = chat_history[-5:]
            for msg in recent_history:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": question})

        # Also return unique document IDs for backward compatibility
        source_ids = list(set(doc.id for doc, _ in relevant_docs))

        # Generate streaming response
        first_chunk = True
        for chunk in self.llm_service.generate_response_stream(
            messages=messages, context=context if relevant_docs else None
        ):
            if first_chunk:
                yield chunk, source_ids, sources
                first_chunk = False
            else:
                yield chunk, [], []

    def _build_context(
        self, 
        relevant_docs: List[Tuple[Document, float]],
        graph_context: Optional[Dict] = None,
    ) -> str:
        """
        Build context string from retrieved documents and graph information.

        Args:
            relevant_docs: List of (document, similarity_score) tuples
            graph_context: Optional graph context with entities and relationships

        Returns:
            Formatted context string with both document content and graph information
        """
        if not relevant_docs:
            return ""

        context_parts = []
        
        # Add document chunks (from ChromaDB vector search)
        context_parts.append("=" * 80)
        context_parts.append("RELEVANT DOCUMENT CHUNKS (from vector similarity search):")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        # Separate vector and graph chunks for better organization
        vector_chunks = []
        graph_chunks = []
        
        for doc, score in relevant_docs:
            source_type = doc.metadata.get("source_type", "vector")
            if source_type == "graph" or doc.metadata.get("from_graph", False):
                graph_chunks.append((doc, score))
            else:
                vector_chunks.append((doc, score))
        
        # Add vector chunks first (primary sources)
        for i, (doc, score) in enumerate(vector_chunks, 1):
            context_parts.append(f"[Source {i}] 📊 Vector (Relevance: {score:.2%})")
            if doc.file_name:
                context_parts.append(f"From: {doc.file_name}")
            chunk_index = doc.metadata.get("chunk_index", 0)
            context_parts.append(f"Chunk Index: {chunk_index}")
            context_parts.append("")
            context_parts.append(doc.content)
            context_parts.append("")  # Empty line between sources
        
        # Add graph chunks (secondary/enrichment sources)
        if graph_chunks:
            context_parts.append("")
            context_parts.append("-" * 80)
            context_parts.append("RELATED CHUNKS (from Graph DB traversal):")
            context_parts.append("-" * 80)
            context_parts.append("")
            
            for i, (doc, score) in enumerate(graph_chunks, 1):
                context_parts.append(f"[Graph Source {i}] 🕸️ Graph (Relevance: {score:.2%})")
                if doc.file_name:
                    context_parts.append(f"From: {doc.file_name}")
                chunk_index = doc.metadata.get("chunk_index", 0)
                context_parts.append(f"Chunk Index: {chunk_index}")
                context_parts.append("")
                context_parts.append(doc.content)
                context_parts.append("")  # Empty line between sources

        # Add graph context (entities and relationships from Neo4j)
        if graph_context:
            context_parts.append("")
            context_parts.append("=" * 80)
            context_parts.append("GRAPH CONTEXT (from Neo4j graph relationships):")
            context_parts.append("=" * 80)
            context_parts.append("")
            
            # Add entities mentioned in chunks
            chunk_entities = graph_context.get("chunk_entities", {})
            if chunk_entities:
                context_parts.append("ENTITIES MENTIONED IN RELEVANT CHUNKS:")
                context_parts.append("-" * 80)
                for chunk_id, entities in chunk_entities.items():
                    if entities:
                        context_parts.append(f"Chunk {chunk_id} mentions:")
                        for entity in entities[:10]:  # Limit to avoid too much context
                            entity_info = f"  - {entity['name']} ({entity['type']})"
                            if entity.get('properties'):
                                props = ", ".join([f"{k}: {v}" for k, v in list(entity['properties'].items())[:3]])
                                if props:
                                    entity_info += f" [{props}]"
                            context_parts.append(entity_info)
                        context_parts.append("")
            
            # Add related entities
            related_entities = graph_context.get("entities", [])
            if related_entities:
                context_parts.append("RELATED ENTITIES (connected through graph relationships):")
                context_parts.append("-" * 80)
                for entity in related_entities[:15]:  # Limit to top related entities
                    entity_info = f"  - {entity['name']} ({entity['type']})"
                    if entity.get('connection_count', 0) > 1:
                        entity_info += f" [connected via {entity['connection_count']} paths]"
                    context_parts.append(entity_info)
                context_parts.append("")
            
            # Add entity relationships
            relationships = graph_context.get("relationships", [])
            if relationships:
                context_parts.append("ENTITY RELATIONSHIPS:")
                context_parts.append("-" * 80)
                for rel in relationships[:20]:  # Limit relationships
                    rel_info = f"  - {rel['source_entity_name']} ({rel['source_entity_type']})"
                    rel_info += f" -[{rel['relationship_type']}]-> "
                    rel_info += f"{rel['target_entity_name']} ({rel['target_entity_type']})"
                    if rel.get('relationship_properties', {}).get('description'):
                        rel_info += f" [{rel['relationship_properties']['description']}]"
                    context_parts.append(rel_info)
                context_parts.append("")
            
            context_parts.append("=" * 80)
            context_parts.append("")

        return "\n".join(context_parts)

    def add_document(self, document: Document) -> None:
        """
        Add a document to the RAG system.
        
        If Graph RAG is enabled, this will also:
        - Extract entities and relationships from chunks
        - Store them in Neo4j graph
        - Link chunks to entities

        Args:
            document: The document to add
        """
        # Save to vector store (ChromaDB)
        self.document_repository.save(document)
        
        # If Graph RAG is enabled, also process for graph
        if self.use_graph_rag and self.hybrid_retrieval_service:
            self._process_document_for_graph(document)

    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the system.

        Returns:
            List of all documents
        """
        return self.document_repository.get_all()

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the system.
        
        If Graph RAG is enabled, this will also delete from Neo4j.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if deleted successfully
        """
        # Delete from vector store
        result = self.document_repository.delete(document_id)
        
        # Delete from graph if Graph RAG is enabled
        if self.use_graph_rag and self.hybrid_retrieval_service:
            neo4j_repo = self.hybrid_retrieval_service.neo4j_repository
            neo4j_repo.delete_document(document_id)
        
        return result
    
    def _process_document_for_graph(self, document: Document) -> None:
        """
        Process a document for Graph RAG: extract entities and store in Neo4j.
        
        Args:
            document: The document to process
        """
        logger.info(f"Processing document for Graph RAG: doc_id={document.id}")
        
        neo4j_repo = self.hybrid_retrieval_service.neo4j_repository
        entity_service = self.hybrid_retrieval_service.entity_extraction_service
        
        # Add document node
        neo4j_repo.add_document(
            document_id=document.id,
            file_name=document.file_name,
            source_type=document.source_type,
        )
        
        # Get chunks from ChromaDB (we need to access the repository's chunks)
        # For now, we'll split the document again to get chunks
        # In a production system, you'd want to reuse the chunks from ChromaDB
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=600,
        )
        chunks = text_splitter.split_text(document.content)
        
        # Process each chunk
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = f"{document.id}_{chunk_index}"
            
            # Add chunk node
            neo4j_repo.add_chunk(
                chunk_id=chunk_id,
                document_id=document.id,
                chunk_index=chunk_index,
                content_preview=chunk_text[:200],
            )
            
            # Extract entities and relationships
            extraction_result = entity_service.extract_entities_and_relationships(
                text=chunk_text,
                chunk_id=chunk_id,
            )
            
            # Store entities and link to chunk
            # Build a map of entity names to IDs for this chunk
            entity_name_to_id = {}
            
            for entity_data in extraction_result.get("entities", []):
                entity_name = entity_data["name"]
                entity_type = entity_data.get("type", "Other")
                
                # Check if entity already exists in Neo4j (entity resolution across chunks)
                existing_entity = neo4j_repo.find_entity_by_name(
                    name=entity_name,
                    entity_type=entity_type,
                )
                
                if existing_entity:
                    # Use existing entity ID
                    entity_id = existing_entity["entity_id"]
                    logger.debug(f"Found existing entity: {entity_name} -> {entity_id}")
                else:
                    # Create new entity
                    entity_id = entity_service.generate_entity_id(
                        name=entity_name,
                        entity_type=entity_type,
                    )
                    neo4j_repo.add_entity(
                        entity_id=entity_id,
                        name=entity_name,
                        entity_type=entity_type,
                        properties=entity_data.get("properties", {}),
                    )
                    logger.debug(f"Created new entity: {entity_name} -> {entity_id}")
                
                # Store in local map for relationship resolution
                entity_name_to_id[entity_name] = entity_id
                
                # Link chunk to entity
                neo4j_repo.link_chunk_to_entity(
                    chunk_id=chunk_id,
                    entity_id=entity_id,
                    confidence=1.0,
                )
            
            # Store relationships between entities
            # Now resolve relationships, looking up entities in Neo4j if not in current chunk
            for rel_data in extraction_result.get("relationships", []):
                source_entity_name = rel_data.get("source_entity_name")
                target_entity_name = rel_data.get("target_entity_name")
                
                if not source_entity_name or not target_entity_name:
                    logger.debug(f"Skipping relationship - missing entity names: {rel_data}")
                    continue
                
                # Try to find source entity ID
                source_entity_id = entity_name_to_id.get(source_entity_name)
                if not source_entity_id:
                    # Look up in Neo4j (might be from another chunk)
                    existing_entity = neo4j_repo.find_entity_by_name(name=source_entity_name)
                    if existing_entity:
                        source_entity_id = existing_entity["entity_id"]
                        logger.debug(f"Resolved source entity from Neo4j: {source_entity_name} -> {source_entity_id}")
                
                # Try to find target entity ID
                target_entity_id = entity_name_to_id.get(target_entity_name)
                if not target_entity_id:
                    # Look up in Neo4j (might be from another chunk)
                    existing_entity = neo4j_repo.find_entity_by_name(name=target_entity_name)
                    if existing_entity:
                        target_entity_id = existing_entity["entity_id"]
                        logger.debug(f"Resolved target entity from Neo4j: {target_entity_name} -> {target_entity_id}")
                
                # If still not found, skip this relationship
                if not source_entity_id or not target_entity_id:
                    logger.debug(
                        f"Skipping relationship - could not resolve entity IDs: "
                        f"source={source_entity_name}, target={target_entity_name}"
                    )
                    continue
                
                # Link entities (create relationship)
                try:
                    neo4j_repo.link_entities(
                        entity1_id=source_entity_id,
                        entity2_id=target_entity_id,
                        relationship_type=rel_data.get("type", "RELATED_TO"),
                        properties={"description": rel_data.get("description", "")},
                    )
                    logger.info(
                        f"Created relationship: {source_entity_name} -[{rel_data.get('type', 'RELATED_TO')}]-> {target_entity_name}"
                    )
                except Exception as e:
                    logger.error(f"Could not link entities: {str(e)}", exc_info=True)
        
        # After processing all chunks, link chunks that mention the same entities
        all_chunk_ids = [f"{document.id}_{i}" for i in range(len(chunks))]
        if len(all_chunk_ids) > 1:
            logger.info(f"Linking chunks that share entities: {len(all_chunk_ids)} chunks from document {document.id}")
            try:
                # Link chunks (both within document and across documents)
                neo4j_repo.link_chunks_sharing_entities(
                    chunk_ids=all_chunk_ids,
                    min_shared_entities=1,  # Link chunks that share at least 1 entity
                    link_across_documents=True,  # Also link to chunks from other documents
                )
            except Exception as e:
                logger.error(f"Error linking chunks: {str(e)}", exc_info=True)
        else:
            logger.debug(f"Skipping chunk linking - only {len(all_chunk_ids)} chunk(s) in document")
        
        logger.info(f"Document processed for Graph RAG: doc_id={document.id}, chunks={len(chunks)}")
