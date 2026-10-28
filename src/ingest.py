from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = Path("data/raw/huggingface_docs.txt")


def load_document():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return file.read()


def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)


if __name__ == "__main__":
    text = load_document()

    chunks = create_chunks(text)

    print(f"Original characters: {len(text)}")
    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- CHUNK {i + 1} ---")
        print(chunk)
