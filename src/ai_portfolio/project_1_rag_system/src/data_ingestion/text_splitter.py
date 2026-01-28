from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TextSplitter:
    """Splits documents into chunks with overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize TextSplitter.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"TextSplitter initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def split_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents from DocumentLoader

        Returns:
            List of document chunks with metadata
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self._split_text(
                doc["content"],
                doc["metadata"]
            )
            chunks.extend(doc_chunks)
            logger.info(
                f"Split '{doc['metadata']['source']}' into {len(doc_chunks)} chunks"
            )
        
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks

    def _split_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split a single document into chunks.

        Args:
            text: Document content
            metadata: Document metadata

        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk_size, save current chunk
            if (
                len(current_chunk) + len(paragraph) > self.chunk_size
                and current_chunk
            ):
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                    },
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                },
            })
        
        return chunks