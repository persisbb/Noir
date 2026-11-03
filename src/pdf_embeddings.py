
from src.onnx_embeddings import embed_texts
from src.pdf_ingest import extract_pdf_text, create_pdf_chunks
from uuid import uuid4
from pathlib import Path



def prepare_pdf_vectors(pdf_path):
    """Extract, chunk, and embed a PDF without uploading it."""
    document_id = uuid4().hex
    pages = extract_pdf_text(pdf_path)
    chunks = create_pdf_chunks(pages)

    if not chunks:
        raise ValueError(
            "No extractable text found in the PDF."
        )

    # Process PDF chunks in small batches to reduce peak memory.
    BATCH_SIZE = 4
    vectors = []

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        batch_vectors = embed_texts(
            [chunk["text"] for chunk in batch]
        )
        vectors.extend(batch_vectors)

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
