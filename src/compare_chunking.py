
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from chunking_v2 import create_chunks_v2


SOURCE_PATH = Path("data/raw/huggingface_docs.txt")
TOP_K = 3
SCORE_THRESHOLD = 0.30
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SECTION_HEADERS = {
    "Hub & Client Libraries",
    "Deployment & Inference",
    "Core ML Libraries",
    "Training & Optimization",
    "Collaboration & Extras",
    "Community",
}

# Same evaluation questions used in the earlier V2 experiment.


TEST_CASES = [
    {
        "question": "What is the Hugging Face Hub?",
        "old_id": "doc-1",
        "new_id": "Hub & Client Libraries — Hub",
    },
    {
        "question": "Where can I host models, datasets, and Spaces?",
        "old_id": "doc-1",
        "new_id": "Hub & Client Libraries — Hub",
    },
    {
        "question": "What does the Hub Python Library do?",
        "old_id": "doc-1",
        "new_id": "Hub & Client Libraries — Hub Python Library",
    },
    {
        "question": "Which tools interact with Hugging Face services?",
        "old_id": "doc-1",
        "new_id": "Hub & Client Libraries — CLI",
    },
    {
        "question": "What are Inference Endpoints?",
        "old_id": "doc-2",
        "new_id": "Deployment & Inference — Inference Endpoints (dedicated)",
    },
    {
        "question": "How can I deploy models on dedicated infrastructure?",
        "old_id": "doc-2",
        "new_id": "Deployment & Inference — Inference Endpoints (dedicated)",
    },
    {
        "question": "What is Text Generation Inference used for?",
        "old_id": "doc-2",
        "new_id": "Deployment & Inference — Text Generation Inference",
    },
    {
        "question": "Which library provides embeddings and reranking?",
        "old_id": "doc-3",
        "new_id": "Core ML Libraries — Sentence Transformers",
    },
    {
        "question": "Which library provides state-of-the-art vision models?",
        "old_id": "doc-3",
        "new_id": "Core ML Libraries — timm",
    },
    {
        "question": "Which library runs machine learning directly in a browser?",
        "old_id": "doc-3",
        "new_id": "Core ML Libraries — Transformers.js",
    },
    {
        "question": "What helps optimize Transformers for faster inference?",
        "old_id": "doc-4",
        "new_id": "Training & Optimization — Optimum",
    },
    {
        "question": "Which tool builds machine learning demos and web apps?",
        "old_id": "doc-5",
        "new_id": "Collaboration & Extras — Gradio",
    },
    {
        "question": "What is PEFT used for?",
        "old_id": "doc-4",
        "new_id": "Training & Optimization — PEFT",
    },
    {
        "question": "What library supports diffusion models in PyTorch?",
        "old_id": "doc-3",
        "new_id": "Core ML Libraries — Diffusers",
    },
    {
        "question": "Which toolkit supports reinforcement learning for transformer language models?",
        "old_id": "doc-4",
        "new_id": "Training & Optimization — TRL",
    },
    {
        "question": "How can I deploy Hugging Face models on Microsoft Azure?",
        "old_id": "doc-2",
        "new_id": "Deployment & Inference — Microsoft Azure",
    },
    {
        "question": "What does the Hugging Face Dataset viewer provide?",
        "old_id": "doc-1",
        "new_id": "Hub & Client Libraries — Dataset viewer",
    },
    {
        "question": "What is the capital of France?",
        "old_id": None,
        "new_id": None,
    },
    {
        "question": "How do I bake a chocolate cake?",
        "old_id": None,
        "new_id": None,
    },
    {
        "question": "Who won the FIFA World Cup in 1998?",
        "old_id": None,
        "new_id": None,
    },
]




def create_original_chunks(text: str) -> list[dict[str, str]]:
    """Re-create the original broad section-based chunks."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    chunks = []
    current_lines = []
    current_section = "Overview"
    chunk_number = 0

    for line in lines:
        if line in SECTION_HEADERS:
            if current_lines:
                chunks.append({
                    "id": f"doc-{chunk_number}",
                    "text": " ".join(current_lines),
                })
                chunk_number += 1

            current_section = line
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append({
            "id": f"doc-{chunk_number}",
            "text": " ".join(current_lines),
        })

    return chunks


def rank_chunks(
    question: str,
    chunks: list[dict[str, str]],
    chunk_vectors: np.ndarray,
    embeddings: HuggingFaceEmbeddings,
) -> list[dict]:
    """Rank chunks by cosine similarity and apply the shared threshold."""
    query_vector = np.array(embeddings.embed_query(question))

    scores = chunk_vectors @ query_vector / (
        np.linalg.norm(chunk_vectors, axis=1)
        * np.linalg.norm(query_vector)
        + 1e-12
    )

    ranked_indices = np.argsort(scores)[::-1][:TOP_K]
    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score >= SCORE_THRESHOLD:
            results.append({
                "id": chunks[index]["id"],
                "score": score,
            })

    return results


def evaluate_strategy(
    strategy_name: str,
    chunks: list[dict[str, str]],
    id_field: str,
    embeddings: HuggingFaceEmbeddings,
) -> dict:
    """Evaluate one chunking strategy on the shared test set."""
    print(f"\n{'=' * 65}")
    print(f"Strategy: {strategy_name}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"{'=' * 65}")

    chunk_vectors = np.array(
        embeddings.embed_documents([chunk["text"] for chunk in chunks])
    )

    recalls = []
    reciprocal_ranks = []
    first_hit_count = 0
    oos_pass_count = 0
    oos_total = 0

    for case in TEST_CASES:
        question = case["question"]
        expected_id = case[id_field]

        results = rank_chunks(
            question, chunks, chunk_vectors, embeddings
        )

        print(f"\nQuestion: {question}")

        for result in results:
            print(
                f"  {result['id']} | "
                f"score={result['score']:.3f}"
            )

        if expected_id is None:
            oos_total += 1

            if not results:
                oos_pass_count += 1
                print("Out-of-scope rejection: PASS")
            else:
                print("Out-of-scope rejection: FAIL")

            continue

        retrieved_ids = [result["id"] for result in results]
        hit_positions = [
            rank + 1
            for rank, chunk_id in enumerate(retrieved_ids)
            if chunk_id == expected_id
        ]

        recall = 1.0 if hit_positions else 0.0
        reciprocal_rank = (
            1.0 / hit_positions[0] if hit_positions else 0.0
        )
        hit_at_1 = bool(hit_positions and hit_positions[0] == 1)

        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        first_hit_count += int(hit_at_1)

        print(f"Expected chunk: {expected_id}")
        print(f"Recall@3: {recall:.2f}")
        print(f"Reciprocal rank: {reciprocal_rank:.2f}")
        print(f"Hit@1: {int(hit_at_1)}")

    summary = {
        "strategy": strategy_name,
        "chunks": len(chunks),
        "recall": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "hit_rate": first_hit_count / len(recalls),
        "oos_pass": oos_pass_count,
        "oos_total": oos_total,
    }

    print(f"\n--- {strategy_name} Summary ---")
    print(f"Mean Recall@3: {summary['recall']:.3f}")
    print(f"MRR: {summary['mrr']:.3f}")
    print(f"Hit Rate@1: {summary['hit_rate']:.3f}")
    print(
        "Out-of-scope rejection: "
        f"{oos_pass_count}/{oos_total}"
    )

    return summary


def main():
    text = SOURCE_PATH.read_text(encoding="utf-8")

    original_chunks = create_original_chunks(text)
    focused_chunks = create_chunks_v2(text)

    print(f"Original broad chunks: {len(original_chunks)}")
    print(f"Focused V2 chunks: {len(focused_chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    original_results = evaluate_strategy(
        "Original broad chunks",
        original_chunks,
        "old_id",
        embeddings,
    )

    focused_results = evaluate_strategy(
        "Focused V2 chunks",
        focused_chunks,
        "new_id",
        embeddings,
    )

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)
    print(
        f"{'Metric':<25}"
        f"{'Original':>15}"
        f"{'Focused V2':>15}"
    )

    metrics = [
        ("Number of chunks", "chunks"),
        ("Mean Recall@3", "recall"),
        ("MRR", "mrr"),
        ("Hit Rate@1", "hit_rate"),
        ("OOS correctly rejected", "oos_pass"),
    ]

    for label, key in metrics:
        old_value = original_results[key]
        new_value = focused_results[key]

        if key in ("recall", "mrr", "hit_rate"):
            old_value = f"{old_value:.3f}"
            new_value = f"{new_value:.3f}"

        print(f"{label:<25}{str(old_value):>15}{str(new_value):>15}")

    print("\nComparison complete. No Pinecone index was modified.")


if __name__ == "__main__":
    main()
