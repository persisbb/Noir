
import os
import sys

from dotenv import load_dotenv
from pinecone import Pinecone

from src.pdf_embeddings import prepare_pdf_vectors


load_dotenv()

INDEX_NAME = "production-rag"
NAMESPACE = "user-documents"


def upload_pdf(pdf_path):
    """Upload PDF vectors without deleting existing data."""

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from .env")

    # Extract text, create chunks, and generate embeddings.
    records = prepare_pdf_vectors(pdf_path)

    # Connect to the existing Pinecone index.
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    # Store uploaded PDFs in a separate namespace.
    # No delete operations are performed.
    index.upsert(
        vectors=records,
        namespace=NAMESPACE,
    )

    print(f"Successfully uploaded {len(records)} vectors!")
    print(f"Index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Document ID: {records[0]['metadata']['document_id']}")


    return {
        "document_id": records[0]["metadata"]["document_id"],
        "filename": records[0]["metadata"]["source"],
        "chunk_count": len(records),
        "namespace": NAMESPACE,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.pdf_pinecone path/to/file.pdf")
        sys.exit(1)

    upload_pdf(sys.argv[1])
