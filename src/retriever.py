import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "production-rag"



# Connect to Pinecone and load the embedding model once
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

from functools import lru_cache


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_retriever():
    return index, get_embeddings()

def search(query, top_k=3):
    index, embeddings = create_retriever()

    # Convert user question into a vector
    query_vector = embeddings.embed_query(query)

    # Search Pinecone
    results = index.query(
    vector=query_vector,
    top_k=top_k,
    include_metadata=True,
    )

    # Keep only sufficiently relevant results
    threshold = 0.30

    filtered_matches = [
        match for match in results["matches"]
        if match["score"] >= threshold
    ]

    results["matches"] = filtered_matches

    return results


def get_index_stats():
    """Read statistics from the existing Pinecone index."""
    stats = index.describe_index_stats()

    return {
        "index_name": INDEX_NAME,
        "total_vectors": stats.total_vector_count,
        "dimension": stats.dimension,
        "namespaces": len(stats.namespaces or {}),
    }


def list_source_chunks():
    """List vector IDs and metadata from the existing index."""
    chunks = []

    # Iterate through the vector IDs in the default namespace.
    for id_batch in index.list(namespace=""):
        ids = list(id_batch)

        if not ids:
            continue

        result = index.fetch(ids=ids, namespace="")

        for vector_id, vector in result.vectors.items():
            metadata = dict(vector.metadata or {})

            chunks.append({
                "id": vector_id,
                "chunk_id": metadata.get("chunk_id", vector_id),
                "preview": str(
                    metadata.get("text", "No text available.")
                )[:160],
            })

    return sorted(chunks, key=lambda item: str(item["chunk_id"]))


def get_source_chunk(vector_id):
    """Fetch a single chunk and its metadata from Pinecone."""
    result = index.fetch(ids=[vector_id], namespace="")

    vector = result.vectors.get(vector_id)

    if vector is None:
        return None

    metadata = dict(vector.metadata or {})

    return {
        "id": vector_id,
        "chunk_id": metadata.get("chunk_id", vector_id),
        "text": metadata.get("text", "No text available."),
        "metadata": metadata,
    }


if __name__ == "__main__":
    test_questions = [
        "What is the Hugging Face Hub?",
        "Which library provides embeddings and reranking?",
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "Who won the FIFA World Cup in 1998?",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")

        # Query Pinecone directly so we can inspect scores
        # before the existing 0.30 threshold filters results.
        query_vector = embeddings.embed_query(question)

        raw_results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True,
        )

        print("Raw matches:")
        for match in raw_results["matches"]:
            print(
                f"  Score: {match['score']:.4f} | "
                f"ID: {match['id']} | "
                f"Text: {match['metadata'].get('text', '')[:120]}"
            )

        filtered_results = search(question)
        print(
            f"Matches remaining after threshold 0.30: "
            f"{len(filtered_results['matches'])}"
        )
