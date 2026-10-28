
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_text(pdf_path):
    """Extract text from each page of a PDF."""

    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip(),
            })

    return pages


def create_pdf_chunks(pages):
    """Split PDF pages into searchable text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_number, chunk_text in enumerate(page_chunks):
            chunks.append({
                "page": page["page"],
                "chunk_number": chunk_number,
                "text": chunk_text,
            })

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.pdf_ingest path/to/file.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    pages = extract_pdf_text(pdf_path)
    chunks = create_pdf_chunks(pages)

    print(f"Pages containing text: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks[:3]:
        print(f"\nPage: {chunk['page']}")
        print(f"Chunk: {chunk['chunk_number']}")
        print(f"Text: {chunk['text'][:300]}")
