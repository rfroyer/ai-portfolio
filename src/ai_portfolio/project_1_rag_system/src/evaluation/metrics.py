"""Metrics Collection - Track evaluation metrics"""
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MetricsSnapshot:
    """Snapshot of evaluation metrics."""
    timestamp: str
    total_queries: int
    average_relevance: float
    average_coherence: float
    average_faithfulness: float
    average_overall: float

class MetricsCollector:
    """Collects and manages evaluation metrics."""
    
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
        logger.info("MetricsCollector initialized")
    
    def add_metric(self, metric: Dict[str, Any]):
        """Add a metric."""
        self.metrics.append({**metric, "timestamp": datetime.now().isoformat()})
    
    def get_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        if not self.metrics:
            return MetricsSnapshot(datetime.now().isoformat(), 0, 0, 0, 0, 0)
        
        avg_relevance = sum(m.get("relevance", 0) for m in self.metrics) / len(self.metrics)
        avg_coherence = sum(m.get("coherence", 0) for m in self.metrics) / len(self.metrics)
        avg_faithfulness = sum(m.get("faithfulness", 0) for m in self.metrics) / len(self.metrics)
        avg_overall = (avg_relevance + avg_coherence + avg_faithfulness) / 3
        
        return MetricsSnapshot(
            timestamp=datetime.now().isoformat(),
            total_queries=len(self.metrics),
            average_relevance=avg_relevance,
            average_coherence=avg_coherence,
            average_faithfulness=avg_faithfulness,
            average_overall=avg_overall,
        )
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        snapshot = self.get_snapshot()
        with open(filepath, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2)
        logger.info(f"Metrics exported to {filepath}")
