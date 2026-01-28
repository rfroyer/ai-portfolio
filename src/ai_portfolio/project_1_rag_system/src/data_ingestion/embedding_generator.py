from typing import List, Dict, Any
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text using OpenAI."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        """
        Initialize EmbeddingGenerator.

        Args:
            model: OpenAI embedding model to use
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)
        logger.info(f"EmbeddingGenerator initialized with model: {model}")

    def generate_embeddings(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of document chunks from TextSplitter

        Returns:
            List of chunks with embeddings
        """
        chunks_with_embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = self._generate_embedding(chunk["content"])
                chunk["embedding"] = embedding
                chunks_with_embeddings.append(chunk)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(chunks)} chunks")
            except Exception as e:
                logger.error(
                    f"Error generating embedding for chunk {i}: {str(e)}"
                )
        
        logger.info(f"Total embeddings generated: {len(chunks_with_embeddings)}")
        return chunks_with_embeddings

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding