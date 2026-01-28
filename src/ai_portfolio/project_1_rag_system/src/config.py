import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Data Paths
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "data", "knowledge_base")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
EVALUATION_DB_PATH = os.path.join(BASE_DIR, "data", "evaluation.db")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 2000

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# RAG Configuration
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.5

# Logging Configuration
LOG_LEVEL = "INFO"

# Verify paths exist
os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

print("Configuration loaded successfully!")