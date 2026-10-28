from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = Path("data/raw/huggingface_docs.txt")


def load_document():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return file.read()


def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    return splitter.split_text(text)


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


if __name__ == "__main__":
    text = load_document()
    chunks = create_chunks(text)

    embeddings = create_embeddings()

    vectors = embeddings.embed_documents(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding dimensions: {len(vectors[0])}")
    print(f"First vector: {vectors[0][:10]}")
