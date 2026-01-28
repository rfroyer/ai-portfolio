import os
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from disk in various formats."""

    SUPPORTED_FORMATS = {".txt", ".md", ".pdf"}

    def __init__(self, knowledge_base_path: str):
        """
        Initialize DocumentLoader.

        Args:
            knowledge_base_path: Path to directory containing documents
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        if not self.knowledge_base_path.exists():
            raise ValueError(
                f"Knowledge base path does not exist: {knowledge_base_path}"
            )
        logger.info(f"DocumentLoader initialized with path: {knowledge_base_path}")

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all supported documents from knowledge base.

        Returns:
            List of documents with metadata
        """
        documents = []
        
        for file_path in self.knowledge_base_path.iterdir():
            if file_path.is_file() and file_path.suffix in self.SUPPORTED_FORMATS:
                try:
                    doc = self._load_single_document(file_path)
                    if doc:
                        documents.append(doc)
                        logger.info(f"Loaded document: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {str(e)}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def _load_single_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a single document based on its format.

        Args:
            file_path: Path to the document

        Returns:
            Document dictionary with content and metadata
        """
        if file_path.suffix == ".txt":
            return self._load_txt(file_path)
        elif file_path.suffix == ".md":
            return self._load_md(file_path)
        elif file_path.suffix == ".pdf":
            return self._load_pdf(file_path)
        return None

    def _load_txt(self, file_path: Path) -> Dict[str, Any]:
        """Load a .txt file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return {
            "content": content,
            "metadata": {
                "source": file_path.name,
                "file_type": "txt",
                "file_path": str(file_path),
            },
        }

    def _load_md(self, file_path: Path) -> Dict[str, Any]:
        """Load a .md (Markdown) file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return {
            "content": content,
            "metadata": {
                "source": file_path.name,
                "file_type": "md",
                "file_path": str(file_path),
            },
        }

    def _load_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load a .pdf file."""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            # Convert PDF to images and extract text
            images = convert_from_path(file_path)
            content = ""
            for image in images:
                content += pytesseract.image_to_string(image) + "\n"
            
            return {
                "content": content,
                "metadata": {
                    "source": file_path.name,
                    "file_type": "pdf",
                    "file_path": str(file_path),
                },
            }
        except ImportError:
            logger.warning(
                "PDF support requires pdf2image and pytesseract. "
                "Install with: pip install pdf2image pytesseract"
            )
            return None