# AutoRAG: Implementation Guide

---

## Document Information

| Field | Value |
|-------|-------|
| **Project Name** | AutoRAG (Autonomous Retrieval-Augmented Generation) |
| **Document Type** | Implementation Guide |
| **Version** | 1.0 |
| **Last Updated** | January 25, 2026 |
| **Status** | Final |

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
mkdir -p data/{knowledge_base,embeddings}
mkdir -p tests
mkdir -p docs
mkdir -p logs
```

### 2.3 Dependencies Installation

Create a `requirements.txt` file with all necessary dependencies:

```
langchain==0.1.0
faiss-cpu==1.7.4
openai==1.3.0
crewai==0.1.0
fastapi==0.104.0
uvicorn==0.24.0
python-dotenv==1.0.0
pytest==7.4.0
python-schedule==0.6.0
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
FAISS_INDEX_PATH=data/embeddings/faiss_index.bin
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

### 3.3 Embedding Generation Module

Create `src/data_ingestion/embeddings.py`:

```python
from openai import OpenAI
import logging
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for text chunks."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model
    
    def generate_embeddings(self, chunks: List[dict]) -> List[dict]:
        """Generate embeddings for all chunks."""
        chunks_with_embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                response = self.client.embeddings.create(
                    input=chunk['content'],
                    model=self.model
                )
                
                embedding = response.data[0].embedding
                chunk['embedding'] = embedding
                chunks_with_embeddings.append(chunk)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(chunks)} chunks")
            
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {i}: {e}")
        
        logger.info(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks")
        return chunks_with_embeddings
```

### 3.4 FAISS Vector Database Module

Create `src/data_ingestion/vector_db.py`:

```python
import faiss
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

class FAISSVectorDB:
    """Manages FAISS vector database."""
    
    def __init__(self, index_path: str, dimension: int = 1536):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.metadata = []
    
    def create_index(self, chunks_with_embeddings: List[dict]) -> None:
        """Create and populate FAISS index."""
        embeddings = np.array([chunk['embedding'] for chunk in chunks_with_embeddings]).astype('float32')
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.metadata = chunks_with_embeddings
        
        logger.info(f"Created FAISS index with {len(embeddings)} vectors")
    
    def save_index(self) -> None:
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(self.index_path))
        
        metadata_path = self.index_path.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved FAISS index to {self.index_path}")
    
    def load_index(self) -> None:
        """Load index from disk."""
        if not self.index_path.exists():
            logger.warning(f"Index not found at {self.index_path}")
            return
        
        self.index = faiss.read_index(str(self.index_path))
        
        metadata_path = self.index_path.with_suffix('.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded FAISS index from {self.index_path}")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[dict]:
        """Search for similar documents."""
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        return results
```

### 3.5 Data Ingestion Pipeline Script

Create `src/data_ingestion/pipeline.py`:

```python
import os
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from embeddings import EmbeddingGenerator
from vector_db import FAISSVectorDB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ingestion_pipeline():
    """Run the complete data ingestion pipeline."""
    
    knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base/")
    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "data/embeddings/faiss_index.bin")
    
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
    
    # Step 3: Generate embeddings
    embedding_gen = EmbeddingGenerator()
    chunks_with_embeddings = embedding_gen.generate_embeddings(chunks)
    
    # Step 4: Create and save FAISS index
    vector_db = FAISSVectorDB(faiss_index_path)
    vector_db.create_index(chunks_with_embeddings)
    vector_db.save_index()
    
    logger.info("Data ingestion pipeline completed successfully!")

if __name__ == "__main__":
    run_ingestion_pipeline()
```

---

## 4. RAG Pipeline Implementation

### 4.1 Retriever Component

Create `src/rag_pipeline/retriever.py`:

```python
from openai import OpenAI
from vector_db import FAISSVectorDB
import logging
from typing import List

logger = logging.getLogger(__name__)

class Retriever:
    """Retrieves relevant documents from vector database."""
    
    def __init__(self, vector_db: FAISSVectorDB):
        self.client = OpenAI()
        self.vector_db = vector_db
        self.embedding_model = "text-embedding-3-small"
    
    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """Retrieve relevant documents for a query."""
        
        # Generate query embedding
        response = self.client.embeddings.create(
            input=query,
            model=self.embedding_model
        )
        query_embedding = response.data[0].embedding
        
        # Search vector database
        results = self.vector_db.search(query_embedding, k=k)
        
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
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
from vector_db import FAISSVectorDB
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Orchestrates the RAG pipeline."""
    
    def __init__(self, vector_db_path: str):
        self.vector_db = FAISSVectorDB(vector_db_path)
        self.vector_db.load_index()
        
        self.retriever = Retriever(self.vector_db)
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
                "sources": []
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGCLI:
    """Command-line interface for RAG system."""
    
    def __init__(self, vector_db_path: str):
        self.rag = RAGPipeline(vector_db_path)
    
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
    import os
    vector_db_path = os.getenv("FAISS_INDEX_PATH", "data/embeddings/faiss_index.bin")
    cli = RAGCLI(vector_db_path)
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

vector_db_path = os.getenv("FAISS_INDEX_PATH", "data/embeddings/faiss_index.bin")
rag_pipeline = RAGPipeline(vector_db_path)

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
    
    def __init__(self, vector_db_path: str, db_path: str, questions_file: str):
        self.rag = RAGPipeline(vector_db_path)
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
    vector_db_path = os.getenv("FAISS_INDEX_PATH", "data/embeddings/faiss_index.bin")
    db_path = os.getenv("EVALUATION_DB_PATH", "data/evaluation_results.db")
    questions_file = "data/evaluation_questions.json"
    
    agent = EvaluationAgent(vector_db_path, db_path, questions_file)
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
    rag = RAGPipeline("data/embeddings/faiss_index.bin")
    assert rag is not None

def test_answer_query():
    """Test query answering."""
    rag = RAGPipeline("data/embeddings/faiss_index.bin")
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
| **FAISS index not found** | Run data ingestion pipeline first |
| **Slow embeddings** | Use smaller chunk sizes or batch processing |
| **Poor retrieval quality** | Adjust chunk size, overlap, or number of results (k) |

---

## 11. Next Steps

1. Complete Phase 1: Environment setup
2. Implement Phase 2: Data ingestion pipeline
3. Implement Phase 3: RAG pipeline core
4. Implement Phase 4: User interfaces
5. Implement Phase 5: Evaluation system
6. Complete Phase 6: Testing and optimization
