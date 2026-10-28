
import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

INDEX_NAME = "production-rag"
NAMESPACE = "user-documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def retrieve_pdf_chunks(question, top_k=3, document_id=None):
    """Retrieve relevant chunks from uploaded PDFs."""

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from .env")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )

    question_vector = embeddings.embed_query(question)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    query_args = {
        "vector": question_vector,
        "top_k": top_k,
        "namespace": NAMESPACE,
        "include_metadata": True,
    }

    if document_id:
        query_args["filter"] = {
            "document_id": {"$eq": document_id}
        }

    results = index.query(**query_args)

    chunks = []

    for match in results.matches:
        metadata = match.metadata or {}

        chunks.append({
            "score": match.score,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page"),
            "document_id": metadata.get("document_id"),
            "chunk_id": metadata.get("chunk_id"),
            "text": metadata.get("text", ""),
        })

    return chunks


def fetch_neighbor_chunks(document_id, chunk_ids):
    """Fetch specific PDF chunks from Pinecone using their IDs."""

    if not document_id or not chunk_ids:
        return []

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from .env")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    vector_ids = [
        f"{document_id}-chunk-{int(chunk_id)}"
        for chunk_id in chunk_ids
        if int(chunk_id) >= 0
    ]

    response = index.fetch(
        ids=vector_ids,
        namespace=NAMESPACE,
    )

    chunks = []

    for vector_id in vector_ids:
        vector = response.vectors.get(vector_id)

        if vector is None:
            continue

        metadata = vector.metadata or {}

        if metadata.get("document_id") != document_id:
            continue

        chunks.append({
            "chunk_id": metadata.get("chunk_id"),
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page"),
            "document_id": metadata.get("document_id"),
            "text": metadata.get("text", ""),
        })

    return chunks


if __name__ == "__main__":
    question = "What are the three types of machine learning?"

    results = retrieve_pdf_chunks(
        question,
        document_id="f1d32e2fe90e43ad96a37f67c778f79d",
    )

    print(f"Question: {question}")
    print(f"Retrieved chunks: {len(results)}")

    for i, chunk in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Similarity: {chunk['score']:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print(f"Text: {chunk['text'][:350]}")
