# AutoRAG: Autonomous Retrieval-Augmented Generation System

A production-ready AI system that combines retrieval and generation to provide accurate, context-aware responses. Built with Python, LangChain, OpenAI, and Chroma vector database.

## 🎯 Overview

AutoRAG is a complete implementation of Retrieval-Augmented Generation (RAG) that demonstrates:

- **Data Ingestion Pipeline**: Loads and processes documents into a vector database
- **RAG Query System**: Retrieves relevant context and generates accurate responses
- **Autonomous Evaluation**: Evaluates system performance with multiple metrics
- **REST API**: Production-ready FastAPI server for deployment
- **CLI Interface**: Command-line tools for easy interaction

## ✨ Key Features

### 📥 Data Ingestion
- Supports multiple file formats (.txt, .md, .pdf)
- Intelligent text chunking with overlap
- OpenAI embeddings generation
- Persistent Chroma vector database

### 🤖 RAG Pipeline
- Semantic similarity search
- Context-aware LLM responses
- Source attribution for transparency
- Token usage tracking

### 📊 Autonomous Evaluation
- Relevance scoring
- Coherence assessment
- Faithfulness evaluation
- Automated metrics collection

### 🌐 Multiple Interfaces
- REST API (FastAPI)
- Command-line interface (Click)
- Interactive chat mode
- Batch query processing

## 🚀 Quick Start

### Prerequisites
- Python 3.11.0
- OpenAI API key
- Virtual environment

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd project_1_rag_system

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key-here"
