"""Query Engine Module - Handles query processing and retrieval"""
from typing import List, Dict, Any
import logging
from openai import OpenAI
from src.data_ingestion.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class QueryEngine:
    """Processes queries and retrieves relevant documents."""
    
    def __init__(self, vector_store: ChromaVectorStore, embedding_model: str = "text-embedding-3-small", api_key: str = None):
        """
        Initialize QueryEngine.

        Args:
            vector_store: ChromaVectorStore instance
            embedding_model: Embedding model to use
            api_key: OpenAI API key
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=api_key)
        logger.info("QueryEngine initialized")

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents with metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self._generate_query_embedding(query)
            
            # Retrieve similar documents from vector store
            retrieved_docs = self.vector_store.retrieve(
                query_embedding=query_embedding,
                n_results=top_k
            )
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
        
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        response = self.client.embeddings.create(
            input=query,
            model=self.embedding_model,
        )
        return response.data[0].embedding

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents as context string.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc["metadata"].get("source", "Unknown")
            similarity = 1 - doc["distance"]  # Convert distance to similarity
            
            context_parts.append(
                f"[Source {i}: {source} (Similarity: {similarity:.2%})]\n"
                f"{doc['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
