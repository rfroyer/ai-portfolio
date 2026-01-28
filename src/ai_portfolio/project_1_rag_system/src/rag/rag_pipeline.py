"""RAG Pipeline Module - Main orchestration"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHROMA_DB_PATH, OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL, TEMPERATURE, TOP_K_RETRIEVAL
from data_ingestion.chroma_store import ChromaVectorStore
from rag.query_engine import QueryEngine
from rag.response_generator import ResponseGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestration."""
    
    def __init__(self, chroma_db_path: str = CHROMA_DB_PATH, embedding_model: str = EMBEDDING_MODEL,
                 llm_model: str = LLM_MODEL, api_key: str = OPENAI_API_KEY, temperature: float = TEMPERATURE,
                 top_k: int = TOP_K_RETRIEVAL):
        self.vector_store = ChromaVectorStore(db_path=chroma_db_path)
        self.query_engine = QueryEngine(vector_store=self.vector_store, embedding_model=embedding_model, api_key=api_key)
        self.response_generator = ResponseGenerator(model=llm_model, api_key=api_key, temperature=temperature)
        self.top_k = top_k
        logger.info("RAGPipeline initialized")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query and return response with sources."""
        try:
            retrieved_docs = self.query_engine.retrieve_context(question, top_k=self.top_k)
            context = self.query_engine.format_context(retrieved_docs)
            response_data = self.response_generator.generate_response(question, context)
            
            return {
                "status": "success",
                "query": question,
                "response": response_data["response"],
                "sources": [doc["metadata"].get("source", "Unknown") for doc in retrieved_docs],
                "usage": response_data.get("usage", {}),
            }
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def interactive_chat(self):
        """Run interactive chat mode."""
        print("\n🤖 AutoRAG Interactive Chat")
        print("=" * 60)
        print("Type your questions below. Type 'exit' to quit.\n")
        
        while True:
            try:
                question = input("You: ").strip()
                if question.lower() == "exit":
                    print("Goodbye!")
                    break
                if not question:
                    continue
                
                result = self.query(question)
                if result["status"] == "success":
                    print(f"\nAssistant: {result['response']}")
                    print(f"\n[Sources: {', '.join(result['sources'])}]")
                    print(f"[Tokens used: {result['usage'].get('total_tokens', 'N/A')}]\n")
                else:
                    print(f"\nError: {result['error']}\n")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}\n")

def main():
    """Main entry point."""
    pipeline = RAGPipeline()
    pipeline.interactive_chat()

if __name__ == "__main__":
    main()
