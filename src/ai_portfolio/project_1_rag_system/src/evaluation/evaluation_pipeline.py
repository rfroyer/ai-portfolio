"""Evaluation Pipeline - Orchestrates evaluation workflow"""
import logging
from typing import List, Dict, Any
from evaluation.evaluation_agent import EvaluationAgent
from evaluation.metrics import MetricsCollector
from rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """Orchestrates evaluation of RAG system."""
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.evaluation_agent = EvaluationAgent()
        self.metrics_collector = MetricsCollector()
        logger.info("EvaluationPipeline initialized")
    
    def evaluate_queries(self, queries: List[str]) -> Dict[str, Any]:
        """Evaluate multiple queries."""
        results = []
        
        for query in queries:
            rag_result = self.rag_pipeline.query(query)
            if rag_result["status"] == "success":
                eval_result = self.evaluation_agent.evaluate_response(
                    query=query,
                    response=rag_result["response"],
                    context="\n".join(rag_result.get("sources", []))
                )
                
                metric = {
                    "query": query,
                    "relevance": eval_result.relevance_score,
                    "coherence": eval_result.coherence_score,
                    "faithfulness": eval_result.faithfulness_score,
                    "overall": eval_result.overall_score,
                }
                self.metrics_collector.add_metric(metric)
                results.append(eval_result)
        
        snapshot = self.metrics_collector.get_snapshot()
        return {
            "evaluated_queries": len(results),
            "metrics": {
                "average_relevance_score": snapshot.average_relevance,
                "average_coherence_score": snapshot.average_coherence,
                "average_faithfulness_score": snapshot.average_faithfulness,
                "average_overall_score": snapshot.average_overall,
            },
            "results": results,
        }
