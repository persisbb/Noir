
from langchain_huggingface import HuggingFaceEmbeddings

from src.pdf_ingest import extract_pdf_text, create_pdf_chunks
from uuid import uuid4
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def prepare_pdf_vectors(pdf_path):
    """Extract, chunk, and embed a PDF without uploading it."""
    document_id = uuid4().hex
    pages = extract_pdf_text(pdf_path)
    chunks = create_pdf_chunks(pages)

    if not chunks:
        raise ValueError(
            "No extractable text found in the PDF."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )

    vectors = embeddings.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    records = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": f"{document_id}-chunk-{i}",
            "values": vector,
            "metadata": {
                "source": Path(pdf_path).name,
                "document_id": document_id,
                "page": chunk["page"],
                "chunk_id": i,
                "text": chunk["text"],
            },
        })

    return records


if __name__ == "__main__":
    records = prepare_pdf_vectors("test_document.pdf")

    print(f"Vectors prepared: {len(records)}")
    print(f"Embedding dimensions: {len(records[0]['values'])}")
    print(f"First vector ID: {records[0]['id']}")
    print(f"First vector page: {records[0]['metadata']['page']}")
