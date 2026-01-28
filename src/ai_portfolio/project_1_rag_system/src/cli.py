"""CLI for AutoRAG"""
import click
from src.rag.rag_pipeline import RAGPipeline
from src.evaluation.evaluation_pipeline import EvaluationPipeline
from src.data_ingestion.pipeline import DataIngestionPipeline

@click.group()
def cli():
    """AutoRAG Command Line Interface"""
    pass

@cli.command()
def ingest():
    """Ingest documents"""
    pipeline = DataIngestionPipeline()
    results = pipeline.run()
    click.echo(f"✓ Ingestion complete: {results}")

@cli.command()
@click.option('-q', '--query', prompt='Enter your question', help='Query text')
def query(query):
    """Query the RAG system"""
    pipeline = RAGPipeline()
    result = pipeline.query(query)
    click.echo(f"Response: {result['response']}")
    click.echo(f"Sources: {', '.join(result['sources'])}")

@cli.command()
def chat():
    """Interactive chat mode"""
    pipeline = RAGPipeline()
    pipeline.interactive_chat()

@cli.command()
@click.option('-q', '--queries', multiple=True, help='Queries to evaluate')
def evaluate(queries):
    """Evaluate RAG system"""
    pipeline = EvaluationPipeline()
    if not queries:
        queries = ["What is RAG?", "How do embeddings work?", "What is semantic search?"]
    results = pipeline.evaluate_queries(queries)
    click.echo(f"✓ Evaluated {results['evaluated_queries']} queries")
    click.echo(f"Average Score: {results['metrics']['average_overall_score']:.2%}")

@cli.command()
def metrics():
    """View evaluation metrics"""
    pipeline = EvaluationPipeline()
    snapshot = pipeline.metrics_collector.get_snapshot()
    click.echo(f"Total Queries: {snapshot.total_queries}")
    click.echo(f"Avg Relevance: {snapshot.average_relevance:.2%}")
    click.echo(f"Avg Coherence: {snapshot.average_coherence:.2%}")
    click.echo(f"Avg Faithfulness: {snapshot.average_faithfulness:.2%}")
    click.echo(f"Overall Score: {snapshot.average_overall:.2%}")

@cli.command()
def version():
    """Show version"""
    click.echo("AutoRAG v1.0.0")

if __name__ == '__main__':
    cli()
