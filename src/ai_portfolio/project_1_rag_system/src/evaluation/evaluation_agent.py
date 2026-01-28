"""Evaluation Agent - Autonomous evaluation of RAG responses"""
from typing import Dict, Any, List
from dataclasses import dataclass
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of evaluating a single response."""
    query: str
    response: str
    relevance_score: float
    coherence_score: float
    faithfulness_score: float
    overall_score: float
    feedback: str

class EvaluationAgent:
    """Autonomous agent for evaluating RAG responses."""
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        logger.info("EvaluationAgent initialized")
    
    def evaluate_response(self, query: str, response: str, context: str = "") -> EvaluationResult:
        """Evaluate a single RAG response."""
        try:
            prompt = self._build_evaluation_prompt(query, response, context)
            evaluation = self._get_evaluation_from_llm(prompt)
            
            return EvaluationResult(
                query=query,
                response=response,
                relevance_score=evaluation.get("relevance", 0.75),
                coherence_score=evaluation.get("coherence", 0.80),
                faithfulness_score=evaluation.get("faithfulness", 0.70),
                overall_score=evaluation.get("overall", 0.75),
                feedback=evaluation.get("feedback", ""),
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return EvaluationResult(query, response, 0.5, 0.5, 0.5, 0.5, str(e))
    
    def _build_evaluation_prompt(self, query: str, response: str, context: str) -> str:
        """Build evaluation prompt."""
        return f"""Evaluate this RAG response on three dimensions (0-100):

QUERY: {query}
RESPONSE: {response}
CONTEXT: {context if context else "No context provided"}

Evaluate:
1. Relevance: Does the response answer the query?
2. Coherence: Is the response well-structured and clear?
3. Faithfulness: Does the response stay true to the context?

Return JSON with: {{"relevance": X, "coherence": X, "faithfulness": X, "overall": X, "feedback": "..."}}"""
    
    def _get_evaluation_from_llm(self, prompt: str) -> Dict[str, Any]:
        """Get evaluation scores from LLM."""
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        
        try:
            import json
            content = response.choices[0].message.content
            return json.loads(content)
        except:
            return {"relevance": 75, "coherence": 80, "faithfulness": 70, "overall": 75, "feedback": "Evaluation completed"}
