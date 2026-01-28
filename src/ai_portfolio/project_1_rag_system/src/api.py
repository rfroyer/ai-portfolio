"""FastAPI REST Server for AutoRAG"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from rag.rag_pipeline import RAGPipeline
from evaluation.evaluation_pipeline import EvaluationPipeline

app = FastAPI(title="AutoRAG API", version="1.0.0")
rag_pipeline = RAGPipeline()
eval_pipeline = EvaluationPipeline()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AutoRAG"}

@app.post("/query")
def query(request: QueryRequest):
    result = rag_pipeline.query(request.query)
    return result

@app.post("/batch-query")
def batch_query(queries: List[str]):
    results = [rag_pipeline.query(q) for q in queries]
    return {"queries": len(results), "results": results}

@app.post("/evaluate")
def evaluate(queries: List[str]):
    results = eval_pipeline.evaluate_queries(queries)
    return results

@app.get("/metrics")
def get_metrics():
    snapshot = eval_pipeline.metrics_collector.get_snapshot()
    return {
        "timestamp": snapshot.timestamp,
        "total_queries": snapshot.total_queries,
        "average_relevance": snapshot.average_relevance,
        "average_coherence": snapshot.average_coherence,
        "average_faithfulness": snapshot.average_faithfulness,
        "average_overall": snapshot.average_overall,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
