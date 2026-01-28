from typing import List, Dict, Any
import logging
import chromadb

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages vector storage and retrieval using Chroma."""

    def __init__(self, db_path: str, collection_name: str = "autorag"):
        """
        Initialize ChromaVectorStore.

        Args:
            db_path: Path to store Chroma database
            collection_name: Name of the collection
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize Chroma client with new API (persistent client)
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(
            f"ChromaVectorStore initialized with db_path={db_path}, "
            f"collection={collection_name}"
        )

    def store_embeddings(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Store embeddings in Chroma.

        Args:
            chunks: List of chunks with embeddings

        Returns:
            Number of embeddings stored
        """
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk['metadata']['source']}_chunk_{chunk['metadata']['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk["content"])
            embeddings.append(chunk["embedding"])
            metadatas.append(chunk["metadata"])
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.info(f"Stored {len(ids)} embeddings in Chroma")
        return len(ids)

    def retrieve(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents from Chroma.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return

        Returns:
            List of similar documents with metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        
        retrieved_docs = []
        if results["documents"] and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                retrieved_docs.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
        
        return retrieved_docs

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "db_path": self.db_path,
        }

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared collection: {self.collection_name}")