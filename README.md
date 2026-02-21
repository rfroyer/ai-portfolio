# AI Portfolio Projects

A comprehensive collection of AI/ML projects demonstrating expertise in:
- Large Language Models (LLMs) and RAG Systems
- Supply Chain Optimization
- Executive Assistant Open Claw


## Projects

### 0. Function Calling
- Reads the user request
- Decides if a tool is needed
- Returns structured arguments

### 1. RAG System
- Retrieval-Augmented Generation implementation
- Document embedding and retrieval
- LLM integration

### 2. Supply Chain Optimization
- Demand forecasting with time-series models
- Inventory optimization
- Cost reduction analysis

### 3. Executive Assistant Open Claw
- Workflow automation system that streamlines the post-deal closure process
- Integrates WhatsApp messaging with enterprise productivity platforms  
- Updates across Monday.com, Asana, and Slack whenever a new deal is closed


## Setup

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run tests
pytest tests/

# Run specific project
python -m ai_portfolio.project_1_rag_system.main
