import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "production-rag"
EMBEDDING_DIMENSION = 384


def load_document():
    path = Path("data/raw/huggingface_docs.txt")

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def create_chunks(text):
    section_headers = [
        "Hub & Client Libraries",
        "Deployment & Inference",
        "Core ML Libraries",
        "Training & Optimization",
        "Collaboration & Extras",
        "Community",
    ]

    lines = text.splitlines()

    chunks = []
    current_chunk = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line in section_headers and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []

        current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def main():

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing from .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print("Creating Pinecone index...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("Waiting for index...")
            time.sleep(2)

    print("Pinecone index ready!")

    index = pc.Index(INDEX_NAME)

# Remove old vectors before uploading the new chunk structure
    index.delete(delete_all=True)
    print("Deleted old vectors.")

    text = load_document()
    chunks = create_chunks(text)

    print(f"Chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectors = embeddings.embed_documents(chunks)

    records = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": f"doc-{i}",
            "values": vector,
            "metadata": {
                "source": "huggingface_docs",
                "chunk_id": i,
                "text": chunk,
            },
        })

    index.upsert(vectors=records)

    print(f"Uploaded {len(records)} vectors to Pinecone!")


if __name__ == "__main__":
    main()