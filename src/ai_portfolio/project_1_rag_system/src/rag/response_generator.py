"""Response Generator Module - Generates LLM responses with context"""
from typing import Dict, Any
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates responses using LLM with retrieved context."""
    
    def __init__(self, model: str = "gpt-4-turbo", api_key: str = None, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)
        logger.info(f"ResponseGenerator initialized with model: {model}")
    
    def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate response using LLM with context."""
        try:
            system_prompt = self._build_system_prompt()
            user_message = self._build_user_message(query, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"response": f"Error: {str(e)}", "model": self.model, "error": str(e)}
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM."""
        return """You are an AI assistant specialized in answering questions about AI, machine learning, and RAG systems.
Your role is to:
1. Answer user questions accurately and comprehensively
2. Use the provided context to ground your responses
3. Cite sources when referencing specific information
4. Be clear and concise in your explanations
5. Acknowledge when you're uncertain or when information is not in the provided context"""
    
    def _build_user_message(self, query: str, context: str) -> str:
        """Build user message with context."""
        return f"""Based on the following context, please answer the user's question.

CONTEXT:
{context}

QUESTION:
{query}

Please provide a comprehensive answer based on the context provided."""
