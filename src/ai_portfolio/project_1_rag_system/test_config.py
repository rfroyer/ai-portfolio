import sys
sys.path.insert(0, 'src')

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    KNOWLEDGE_BASE_PATH,
    EVALUATION_DB_PATH,
    LOG_LEVEL
)

print("✓ Configuration Test Results:")
print(f"  OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'NOT SET'}")
print(f"  OPENAI_MODEL: {OPENAI_MODEL}")
print(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"  CHROMA_PERSIST_DIR: {CHROMA_PERSIST_DIR}")
print(f"  KNOWLEDGE_BASE_PATH: {KNOWLEDGE_BASE_PATH}")
print(f"  EVALUATION_DB_PATH: {EVALUATION_DB_PATH}")
print(f"  LOG_LEVEL: {LOG_LEVEL}")
print("\n✓ All configuration loaded successfully!")