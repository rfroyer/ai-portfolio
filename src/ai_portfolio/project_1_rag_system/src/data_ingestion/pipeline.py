import logging
import sys
from pathlib import Path
from typing import Dict, Any  # ← ADD THIS LINE

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    KNOWLEDGE_BASE_PATH,
    CHROMA_DB_PATH,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
)
from data_ingestion.document_loader import DocumentLoader
from data_ingestion.text_splitter import TextSplitter
from data_ingestion.embedding_generator import EmbeddingGenerator
from data_ingestion.chroma_store import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Orchestrates the complete data ingestion workflow."""

    def __init__(
        self,
        knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
        chroma_db_path: str = CHROMA_DB_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        api_key: str = OPENAI_API_KEY,
    ):
        """
        Initialize the pipeline.

        Args:
            knowledge_base_path: Path to documents
            chroma_db_path: Path to Chroma database
            embedding_model: Embedding model to use
            api_key: OpenAI API key
        """
        self.knowledge_base_path = knowledge_base_path
        self.chroma_db_path = chroma_db_path
        
        # Initialize components
        self.document_loader = DocumentLoader(knowledge_base_path)
        self.text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_generator = EmbeddingGenerator(
            model=embedding_model,
            api_key=api_key,
        )
        self.vector_store = ChromaVectorStore(
            db_path=chroma_db_path,
            collection_name="autorag",
        )
        
        logger.info("DataIngestionPipeline initialized")

    def run(self) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline.

        Returns:
            Pipeline execution results
        """
        logger.info("=" * 60)
        logger.info("Starting Data Ingestion Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load documents
            logger.info("\n[Step 1/4] Loading documents...")
            documents = self.document_loader.load_documents()
            if not documents:
                logger.error("No documents loaded!")
                return {"status": "failed", "error": "No documents loaded"}
            
            # Step 2: Split documents
            logger.info("\n[Step 2/4] Splitting documents into chunks...")
            chunks = self.text_splitter.split_documents(documents)
            if not chunks:
                logger.error("No chunks created!")
                return {"status": "failed", "error": "No chunks created"}
            
            # Step 3: Generate embeddings
            logger.info("\n[Step 3/4] Generating embeddings...")
            chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
            if not chunks_with_embeddings:
                logger.error("No embeddings generated!")
                return {"status": "failed", "error": "No embeddings generated"}
            
            # Step 4: Store in vector database
            logger.info("\n[Step 4/4] Storing embeddings in Chroma...")
            num_stored = self.vector_store.store_embeddings(chunks_with_embeddings)
            
            # Get collection stats
            stats = self.vector_store.get_collection_stats()
            
            logger.info("\n" + "=" * 60)
            logger.info("✓ Data Ingestion Pipeline Completed Successfully!")
            logger.info("=" * 60)
            logger.info(f"Documents loaded: {len(documents)}")
            logger.info(f"Chunks created: {len(chunks)}")
            logger.info(f"Embeddings generated: {len(chunks_with_embeddings)}")
            logger.info(f"Embeddings stored: {num_stored}")
            logger.info(f"Collection stats: {stats}")
            logger.info("=" * 60 + "\n")
            
            return {
                "status": "success",
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "embeddings_generated": len(chunks_with_embeddings),
                "embeddings_stored": num_stored,
                "stats": stats,
            }
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {"status": "failed", "error": str(e)}


def main():
    """Main entry point for the pipeline."""
    pipeline = DataIngestionPipeline()
    results = pipeline.run()
    
    if results["status"] == "failed":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()