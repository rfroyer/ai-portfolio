# AutoRAG: Implementation Guide

---

## Document Information

| Field | Value |
|-------|-------|
| **Project Name** | AutoRAG (Autonomous Retrieval-Augmented Generation) |
| **Document Type** | Implementation Guide |
| **Version** | 1.1 |
| **Last Updated** | January 25, 2026 |
| **Status** | Final |
| **Changes** | Updated from FAISS to Chroma for vector database |

---

## 1. Implementation Overview

This document provides a detailed, step-by-step implementation guide for building the AutoRAG system as defined in the architecture. It covers all technical implementation details, code structure, configuration, and best practices for each component of the system.

---

## 2. Environment Setup

### 2.1 Prerequisites

Before beginning implementation, ensure you have the following installed on your system:

*   Python 3.10 or higher
*   pip (Python package manager)
*   Git for version control
*   A text editor or IDE (Visual Studio Code recommended)
*   An OpenAI API key with access to GPT-4 and Embeddings models

### 2.2 Project Initialization

Create the project directory structure and initialize the Python environment:

```bash
# Create project directory
mkdir -p project_1_rag_system
cd project_1_rag_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create project structure
mkdir -p src/{data_ingestion,rag_pipeline,evaluation,ui}
mkdir -p data/{knowledge_base,chroma_db}
mkdir -p tests
mkdir -p docs
mkdir -p logs
```

### 2.3 Dependencies Installation

Create a `requirements.txt` file with all necessary dependencies:

```
langchain>=0.1.0
chromadb>=0.5.0
openai>=1.0.0
crewai>=0.1.0
fastapi>=0.100.0
uvicorn>=0.24.0
python-dotenv>=1.0.0
pytest>=7.0.0
schedule>=1.2.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2.4 Configuration Setup

Create a `.env` file in the project root to store API keys and configuration:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=data/chroma_db
KNOWLEDGE_BASE_PATH=data/knowledge_base/
EVALUATION_DB_PATH=data/evaluation_results.db
LOG_LEVEL=INFO
```

Load environment variables in your Python code:

```python
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
```

---

## 3. Data Ingestion Pipeline Implementation

### 3.1 Document Loader Module

Create `src/data_ingestion/document_loader.py`:

```python
import os
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loads documents from a directory."""
    
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.supported_formats = {'.txt', '.md', '.pdf'}
    
    def load_documents(self) -> List[dict]:
        """Load all documents from knowledge base."""
        documents = []
        
        if not self.knowledge_base_path.exists():
            logger.warning(f"Knowledge base path not found: {self.knowledge_base_path}")
            return documents
        
        for file_path in self.knowledge_base_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.supported_formats:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    documents.append({
                        'content': content,
                        'source': str(file_path),
                        'filename': file_path.name
                    })
                    logger.info(f"Loaded document: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
```

### 3.2 Text Splitter Module

Create `src/data_ingestion/text_splitter.py`:

```python
from typing import List
import logging

logger = logging.getLogger(__name__)

class TextSplitter:
    """Splits documents into chunks for embedding."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_documents(self, documents: List[dict]) -> List[dict]:
        """Split documents into chunks."""
        chunks = []
        
        for doc in documents:
            content = doc['content']
            words = content.split()
            
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunks.append({
                    'content': chunk_text,
                    'source': doc['source'],
                    'filename': doc['filename'],
                    'chunk_index': len([c for c in chunks if c['source'] == doc['source']])
                })
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
```

### 3.3 Chroma Vector Database Module

Create `src/data_ingestion/vector_db.py`:

```python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

class ChromaVectorDB:
    """Manages Chroma vector database with LangChain integration."""
    
    def __init__(self, persist_dir: str, collection_name: str = "autorag_documents"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize or load Chroma vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        
        logger.info(f"Initialized Chroma vector store at {self.persist_dir}")
    
    def add_documents(self, chunks: List[dict]) -> None:
        """Add document chunks to Chroma vector store."""
        try:
            # Prepare documents for Chroma
            texts = [chunk['content'] for chunk in chunks]
            metadatas = [
                {
                    'source': chunk['source'],
                    'filename': chunk['filename'],
                    'chunk_index': chunk['chunk_index']
                }
                for chunk in chunks
            ]
            
            # Add to Chroma
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            self.vectorstore.persist()
            
            logger.info(f"Added {len(chunks)} chunks to Chroma vector store")
        
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[dict]:
        """Search for similar documents in Chroma."""
        try:
            results = self.vectorstore.similarity_search_with_scores(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'similarity_score': score
                })
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching Chroma: {e}")
            return []
    
    def get_retriever(self, k: int = 5):
        """Get LangChain retriever from Chroma."""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
```

### 3.4 Data Ingestion Pipeline Script

Create `src/data_ingestion/pipeline.py`:

```python
import os
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from vector_db import ChromaVectorDB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ingestion_pipeline():
    """Run the complete data ingestion pipeline."""
    
    knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base/")
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    
    logger.info("Starting data ingestion pipeline...")
    
    # Step 1: Load documents
    loader = DocumentLoader(knowledge_base_path)
    documents = loader.load_documents()
    
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return
    
    # Step 2: Split documents
    splitter = TextSplitter(chunk_size=512, overlap=50)
    chunks = splitter.split_documents(documents)
    
    # Step 3: Create Chroma vector store and add documents
    vector_db = ChromaVectorDB(chroma_persist_dir)
    vector_db.add_documents(chunks)
    
    logger.info("Data ingestion pipeline completed successfully!")

if __name__ == "__main__":
    run_ingestion_pipeline()
```

---

## 4. RAG Pipeline Implementation

### 4.1 Retriever Component

Create `src/rag_pipeline/retriever.py`:

```python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import logging
from typing import List

logger = logging.getLogger(__name__)

class Retriever:
    """Retrieves relevant documents using Chroma."""
    
    def __init__(self, persist_dir: str):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        self.vectorstore = Chroma(
            collection_name="autorag_documents",
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
    
    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """Retrieve relevant documents for a query."""
        
        try:
            results = self.vectorstore.similarity_search_with_scores(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'similarity_score': score
                })
            
            logger.info(f"Retrieved {len(formatted_results)} documents for query: {query[:50]}...")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
```

### 4.2 Generator Component

Create `src/rag_pipeline/generator.py`:

```python
from openai import OpenAI
import logging
from typing import List

logger = logging.getLogger(__name__)

class Generator:
    """Generates responses using LLM with context."""
    
    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model
    
    def generate(self, query: str, context: List[dict]) -> str:
        """Generate response based on query and context."""
        
        # Build context string
        context_text = "\n\n".join([
            f"Source: {doc['source']}\n{doc['content'][:500]}..."
            for doc in context
        ])
        
        # Create prompt
        prompt = f"""You are a helpful assistant. Answer the following question based on the provided context.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated response for query: {query[:50]}...")
            return answer
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I was unable to generate a response."
```

### 4.3 RAG Pipeline Orchestrator

Create `src/rag_pipeline/rag.py`:

```python
from retriever import Retriever
from generator import Generator
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Orchestrates the RAG pipeline with Chroma."""
    
    def __init__(self, chroma_persist_dir: str):
        self.retriever = Retriever(chroma_persist_dir)
        self.generator = Generator()
    
    def answer_query(self, query: str, k: int = 5) -> dict:
        """Answer a query using RAG pipeline."""
        
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents
        context = self.retriever.retrieve(query, k=k)
        
        if not context:
            return {
                "query": query,
                "answer": "I could not find relevant information to answer your question.",
                "sources": [],
                "context_count": 0
            }
        
        # Generate response
        answer = self.generator.generate(query, context)
        
        # Compile sources
        sources = list(set([doc['source'] for doc in context]))
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "context_count": len(context)
        }
```

---

## 5. User Interface Implementation

### 5.1 CLI Interface

Create `src/ui/cli.py`:

```python
import argparse
import sys
from rag_pipeline import RAGPipeline
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGCLI:
    """Command-line interface for RAG system."""
    
    def __init__(self, chroma_persist_dir: str):
        self.rag = RAGPipeline(chroma_persist_dir)
    
    def run(self):
        """Run CLI interface."""
        parser = argparse.ArgumentParser(
            description="AutoRAG: Autonomous Retrieval-Augmented Generation System"
        )
        
        parser.add_argument(
            "query",
            type=str,
            help="Question to ask the RAG system"
        )
        
        parser.add_argument(
            "-k",
            "--context-count",
            type=int,
            default=5,
            help="Number of context documents to retrieve (default: 5)"
        )
        
        args = parser.parse_args()
        
        # Get answer
        result = self.rag.answer_query(args.query, k=args.context_count)
        
        # Display result
        print("\n" + "="*60)
        print(f"Query: {result['query']}")
        print("="*60)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources ({len(result['sources'])}):")
        for source in result['sources']:
            print(f"  - {source}")
        print("="*60 + "\n")

if __name__ == "__main__":
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    cli = RAGCLI(chroma_persist_dir)
    cli.run()
```

### 5.2 FastAPI Interface

Create `src/ui/api.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AutoRAG API",
    description="Autonomous Retrieval-Augmented Generation System",
    version="1.0.0"
)

chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
rag_pipeline = RAGPipeline(chroma_persist_dir)

class QueryRequest(BaseModel):
    query: str
    context_count: int = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list
    context_count: int

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Submit a query to the RAG system."""
    try:
        result = rag_pipeline.answer_query(request.query, k=request.context_count)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 6. Autonomous Evaluation System Implementation

### 6.1 Evaluation Database

Create `src/evaluation/evaluation_db.py`:

```python
import sqlite3
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EvaluationDatabase:
    """Manages evaluation results database."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                accuracy_score REAL,
                relevance_score REAL,
                overall_score REAL,
                feedback TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized evaluation database at {self.db_path}")
    
    def save_evaluation(self, question: str, answer: str, scores: dict, feedback: str = None):
        """Save evaluation result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO evaluations 
            (timestamp, question, answer, accuracy_score, relevance_score, overall_score, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            question,
            answer,
            scores.get('accuracy', 0),
            scores.get('relevance', 0),
            scores.get('overall', 0),
            feedback
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved evaluation for question: {question[:50]}...")
```

### 6.2 Evaluation Logic

Create `src/evaluation/evaluation_logic.py`:

```python
from openai import OpenAI
import json
import logging

logger = logging.getLogger(__name__)

class EvaluationLogic:
    """Evaluates RAG system responses."""
    
    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model
    
    def evaluate(self, question: str, answer: str, ground_truth: str = None) -> dict:
        """Evaluate response quality."""
        
        prompt = f"""Evaluate the following response to a question on a scale of 0-10.

Question: {question}
Answer: {answer}
"""
        
        if ground_truth:
            prompt += f"Expected Answer: {ground_truth}\n"
        
        prompt += """
Provide scores for:
1. Accuracy (0-10): How accurate is the answer?
2. Relevance (0-10): How relevant is the answer to the question?

Respond in JSON format:
{
    "accuracy": <score>,
    "relevance": <score>,
    "overall": <average score>,
    "feedback": "<brief explanation>"
}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.choices[0].message.content
            scores = json.loads(response_text)
            
            logger.info(f"Evaluated question: {question[:50]}... - Overall: {scores['overall']}")
            return scores
        
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {"accuracy": 0, "relevance": 0, "overall": 0, "feedback": "Evaluation error"}
```

### 6.3 Evaluation Agent

Create `src/evaluation/evaluation_agent.py`:

```python
import json
import logging
from evaluation_logic import EvaluationLogic
from evaluation_db import EvaluationDatabase
from rag_pipeline import RAGPipeline
import os

logger = logging.getLogger(__name__)

class EvaluationAgent:
    """Autonomous evaluation agent."""
    
    def __init__(self, chroma_persist_dir: str, db_path: str, questions_file: str):
        self.rag = RAGPipeline(chroma_persist_dir)
        self.eval_logic = EvaluationLogic()
        self.eval_db = EvaluationDatabase(db_path)
        self.questions = self.load_questions(questions_file)
    
    def load_questions(self, questions_file: str) -> list:
        """Load evaluation questions."""
        try:
            with open(questions_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            return []
    
    def run_evaluation(self):
        """Run autonomous evaluation."""
        logger.info(f"Starting evaluation with {len(self.questions)} questions...")
        
        for i, q in enumerate(self.questions):
            question = q['question']
            ground_truth = q.get('ground_truth', None)
            
            # Get RAG answer
            result = self.rag.answer_query(question)
            answer = result['answer']
            
            # Evaluate answer
            scores = self.eval_logic.evaluate(question, answer, ground_truth)
            
            # Save to database
            self.eval_db.save_evaluation(question, answer, scores)
            
            logger.info(f"Evaluated {i+1}/{len(self.questions)} questions")
        
        logger.info("Evaluation completed!")

if __name__ == "__main__":
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    db_path = os.getenv("EVALUATION_DB_PATH", "data/evaluation_results.db")
    questions_file = "data/evaluation_questions.json"
    
    agent = EvaluationAgent(chroma_persist_dir, db_path, questions_file)
    agent.run_evaluation()
```

---

## 7. Testing

### 7.1 Unit Tests

Create `tests/test_rag_pipeline.py`:

```python
import pytest
from src.rag_pipeline.rag import RAGPipeline

def test_rag_pipeline_initialization():
    """Test RAG pipeline initialization."""
    rag = RAGPipeline("data/chroma_db")
    assert rag is not None

def test_answer_query():
    """Test query answering."""
    rag = RAGPipeline("data/chroma_db")
    result = rag.answer_query("What is AI?")
    
    assert "query" in result
    assert "answer" in result
    assert "sources" in result
```

---

## 8. Deployment

### 8.1 Running the Data Ingestion

```bash
cd src/data_ingestion
python pipeline.py
```

### 8.2 Running the CLI

```bash
cd src/ui
python cli.py "What is machine learning?"
```

### 8.3 Running the API

```bash
cd src/ui
python -m uvicorn api:app --reload
```

Access the API at `http://localhost:8000` and view documentation at `http://localhost:8000/docs`.

---

## 9. Monitoring and Logging

All modules use Python's built-in logging. Configure logging in your main script:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

---

## 10. Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **OpenAI API errors** | Verify API key is set correctly in .env file |
| **Chroma database not found** | Run data ingestion pipeline first to create embeddings |
| **Slow embeddings** | Use smaller chunk sizes or batch processing |
| **Poor retrieval quality** | Adjust chunk size, overlap, or number of results (k) |
| **Chroma persistence issues** | Verify `CHROMA_PERSIST_DIR` has write permissions |

---

## 11. Chroma-Specific Configuration

### Collection Management

```python
# Create or access a specific collection
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db"
)

# Get collection info
collection = vectorstore._collection
print(f"Collection count: {collection.count()}")
```

### Advanced Filtering

```python
# Search with metadata filtering
results = vectorstore.similarity_search_with_score(
    query="machine learning",
    k=5,
    where={"filename": {"$eq": "ai_guide.txt"}}
)
```

### Persistence

```python
# Manually persist changes
vectorstore.persist()

# Load existing collection
vectorstore = Chroma(
    collection_name="autorag_documents",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db"
)
```

---

## 12. Next Steps

1. Complete Phase 1: Environment setup
2. Implement Phase 2: Data ingestion pipeline with Chroma
3. Implement Phase 3: RAG pipeline core
4. Implement Phase 4: User interfaces
5. Implement Phase 5: Evaluation system
6. Complete Phase 6: Testing and optimization
