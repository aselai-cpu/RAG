"""
ChromaDB Implementation of Document Repository.

This is the infrastructure layer implementation that adapts ChromaDB
to our domain repository interface. This represents the Anti-Corruption Layer
pattern - translating between our domain model and the external ChromaDB API.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Optional
import time
from src.domain.entities.document import Document
from src.domain.repositories.document_repository import IDocumentRepository
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.infrastructure.logging import RAGLogger

# Get logger for ChromaDB operations
logger = RAGLogger.get_logger('chromadb_ops')


class ChromaDocumentRepository(IDocumentRepository):
    """
    ChromaDB implementation of the document repository.

    This class handles the complexity of chunking documents, managing
    embeddings, and interacting with ChromaDB while presenting a simple
    domain-aligned interface.
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./data/chroma",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the ChromaDB repository.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # In-memory cache of document metadata
        self._document_cache: dict[str, Document] = {}

    def save(self, document: Document) -> None:
        """
        Save a document by chunking it and storing in ChromaDB.

        The document is split into chunks to improve retrieval quality.
        Each chunk is stored with metadata linking back to the original document.
        """
        start_time = time.time()
        logger.info(f"Starting document save: doc_id={document.id}, source_type={document.source_type}, file_name={document.file_name}")

        try:
            # Split document into chunks
            chunks = self.text_splitter.split_text(document.content)
            document.chunk_count = len(chunks)
            logger.debug(f"Document chunked: doc_id={document.id}, chunk_count={len(chunks)}, content_length={len(document.content)}")

            # Prepare data for ChromaDB
            chunk_ids = [f"{document.id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_id": document.id,
                    "chunk_index": i,
                    "source_type": document.source_type,
                    "file_name": document.file_name or "",
                    "created_at": document.created_at.isoformat(),
                    **document.metadata,
                }
                for i in range(len(chunks))
            ]

            # Add to ChromaDB
            logger.debug(f"Adding {len(chunks)} chunks to ChromaDB for doc_id={document.id}")
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=chunk_ids,
            )

            # Cache the document metadata
            self._document_cache[document.id] = document

            elapsed = time.time() - start_time
            logger.info(f"Document saved successfully: doc_id={document.id}, chunks={len(chunks)}, time={elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error saving document: doc_id={document.id}, error={str(e)}, time={elapsed:.2f}s", exc_info=True)
            raise

    def get_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.

        Note: This reconstructs the document from its chunks.
        """
        if document_id in self._document_cache:
            return self._document_cache[document_id]

        # Query ChromaDB for all chunks of this document
        results = self.collection.get(
            where={"document_id": document_id},
        )

        if not results["documents"]:
            return None

        # Reconstruct document from chunks
        # Sort chunks by index
        chunks_with_metadata = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1]["chunk_index"],
        )

        content = "\n".join(chunk for chunk, _ in chunks_with_metadata)
        first_metadata = chunks_with_metadata[0][1]

        document = Document(
            id=document_id,
            content=content,
            source_type=first_metadata["source_type"],
            file_name=first_metadata.get("file_name") or None,
            chunk_count=len(chunks_with_metadata),
        )

        self._document_cache[document_id] = document
        return document

    def get_all(self) -> List[Document]:
        """
        Retrieve all documents.

        Note: This can be expensive for large collections.
        """
        # Get all unique document IDs from the collection
        all_results = self.collection.get()

        if not all_results["metadatas"]:
            return []

        # Extract unique document IDs
        document_ids = set(
            metadata["document_id"] for metadata in all_results["metadatas"]
        )

        # Get each document
        documents = []
        for doc_id in document_ids:
            doc = self.get_by_id(doc_id)
            if doc:
                documents.append(doc)

        return documents

    def delete(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id},
            )

            if not results["ids"]:
                return False

            # Delete all chunks
            self.collection.delete(ids=results["ids"])

            # Remove from cache
            self._document_cache.pop(document_id, None)

            return True
        except Exception:
            return False

    def search_similar(
        self, query: str, top_k: int = 5
    ) -> List[tuple[Document, float]]:
        """
        Search for document chunks similar to the query.

        Returns documents with their similarity scores.
        The same document may appear multiple times if multiple chunks match.
        """
        start_time = time.time()
        logger.info(f"Starting similarity search: query_length={len(query)}, top_k={top_k}, query_preview='{query[:100]}...'")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
            )

            if not results["documents"] or not results["documents"][0]:
                logger.info(f"No results found for query")
                return []

            # Extract results
            chunks = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            # Convert distances to similarity scores (1 - distance for cosine)
            similarities = [1 - dist for dist in distances]

            logger.debug(f"Raw search results: num_results={len(chunks)}, distances={distances}, similarities={similarities}")

            # Group by document and aggregate scores
            doc_results = []
            for chunk, metadata, similarity in zip(chunks, metadatas, similarities):
                doc_id = metadata["document_id"]

                # Create a partial document with the matching chunk
                partial_doc = Document(
                    id=doc_id,
                    content=chunk,  # Just the matching chunk
                    source_type=metadata["source_type"],
                    file_name=metadata.get("file_name") or None,
                    metadata={"chunk_index": metadata["chunk_index"]},
                )

                doc_results.append((partial_doc, similarity))
                logger.debug(f"Result: doc_id={doc_id}, chunk_index={metadata['chunk_index']}, similarity={similarity:.4f}, chunk_preview='{chunk[:100]}...'")

            elapsed = time.time() - start_time
            logger.info(f"Search completed: num_results={len(doc_results)}, time={elapsed:.2f}s")

            return doc_results

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in similarity search: error={str(e)}, time={elapsed:.2f}s", exc_info=True)
            raise
