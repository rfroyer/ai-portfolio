# Vector Embeddings and Similarity Search

## What Are Embeddings?
Embeddings are numerical representations of text, images, or other data. 
They capture semantic meaning in a high-dimensional space.

## Text Embeddings
- Convert text to vectors (e.g., 1536 dimensions for text-embedding-3-small)
- Similar texts have similar vectors
- Enable semantic search and clustering

## Popular Embedding Models
| Model | Dimensions | Use Case |
|-------|-----------|----------|
| text-embedding-3-small | 1536 | Fast, efficient, general purpose |
| text-embedding-3-large | 3072 | High accuracy, resource intensive |
| all-MiniLM-L6-v2 | 384 | Lightweight, local deployment |

## Similarity Metrics
- **Cosine Similarity**: Measures angle between vectors (0-1)
- **Euclidean Distance**: Measures straight-line distance
- **Dot Product**: Fast similarity computation

## Vector Databases
Store and query embeddings efficiently:
- Chroma: Lightweight, local-first
- Pinecone: Managed, scalable
- Weaviate: Open-source, flexible
- Milvus: High-performance

## RAG Pipeline Integration
1. Embed documents during ingestion
2. Store embeddings in vector database
3. Embed user query at runtime
4. Retrieve similar documents
5. Pass to LLM for generation

## Best Practices
- Use consistent embedding model across pipeline
- Chunk documents appropriately (500-1000 tokens)
- Normalize embeddings for better similarity
- Monitor embedding quality and relevance
