
from pathlib import Path

SECTION_HEADERS = {
    "Hub & Client Libraries",
    "Deployment & Inference",
    "Core ML Libraries",
    "Training & Optimization",
    "Collaboration & Extras",
    "Community",
}



def create_chunks_v2(text: str) -> list[dict[str, str]]:
    """Create focused chunks containing a section, item, and description."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks = []
    section = "Overview"

    # The opening lines describe the documentation as a whole.
    intro_lines = []

    i = 0
    while i < len(lines) and lines[i] not in SECTION_HEADERS:
        intro_lines.append(lines[i])
        i += 1

    if intro_lines:
        chunks.append({
            "id": "Overview",
            "text": " ".join(intro_lines),
        })

    while i < len(lines):
        if lines[i] in SECTION_HEADERS:
            section = lines[i]
            i += 1
            continue

        item = lines[i]
        i += 1

        # Community entries are names without descriptions.
        if section == "Community":
            chunks.append({
                "id": f"{section} — {item}",
                "text": f"Section: {section}. Resource: {item}.",
            })
            continue

        # Other sections use item-name followed by its description.
        description = ""
        if i < len(lines) and lines[i] not in SECTION_HEADERS:
            description = lines[i]
            i += 1

        # Build the text for this individual chunk.
        chunk_text = (
            f"Section: {section}. "
            f"{item}: {description}"
        )

        # Improve retrieval for natural-language questions about the Hub.

        # Improve retrieval for common natural-language questions.
        if section == "Hub & Client Libraries" and item == "Hub":
            chunk_text = (
                f"Section: {section}. "
                "The Hugging Face Hub is a platform to host "
                "Git-based models, datasets, and Spaces."
            )

        elif section == "Training & Optimization" and item == "PEFT":
            chunk_text = (
                f"Section: {section}. "
                "PEFT (Parameter-Efficient Fine-Tuning) helps fine-tune "
                "large language models by updating a small number of "
                "parameters instead of all model parameters. "
                "It reduces the trainable parameters and resource "
                "requirements of fine-tuning."
            )

        elif section == "Hub & Client Libraries" and item == "Dataset viewer":
            chunk_text = (
                f"Section: {section}. "
                "The Hugging Face Dataset viewer provides an API to "
                "access metadata, statistics, and content for datasets "
                "hosted on the Hugging Face Hub."
            )


        chunks.append({
            "id": f"{section} — {item}",
            "text": chunk_text,
        })

    return chunks



if __name__ == "__main__":
    source_path = Path("data/raw/huggingface_docs.txt")
    source_text = source_path.read_text(encoding="utf-8")
    chunks = create_chunks_v2(source_text)

    print(f"Created {len(chunks)} focused chunks.\n")

    for chunk in chunks:
        print(f"ID: {chunk['id']}")
        print(f"Text: {chunk['text']}\n")
