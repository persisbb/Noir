
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from chunking_v2 import create_chunks_v2


SOURCE_PATH = Path("data/raw/huggingface_docs.txt")
TOP_K = 3
SCORE_THRESHOLD = 0.30

TEST_CASES = [
    {
        "question": "What is the Hugging Face Hub?",
        "relevant_ids": ["Hub & Client Libraries — Hub"],
    },
    {
        "question": "Where can I host models, datasets, and Spaces?",
        "relevant_ids": ["Hub & Client Libraries — Hub"],
    },
    {
        "question": "What does the Hub Python Library do?",
        "relevant_ids": ["Hub & Client Libraries — Hub Python Library"],
    },
    {
        "question": "Which tools interact with Hugging Face services?",
        "relevant_ids": ["Hub & Client Libraries — CLI"],
    },
    {
        "question": "What are Inference Endpoints?",
        "relevant_ids": [
            "Deployment & Inference — Inference Endpoints (dedicated)"
        ],
    },
    {
        "question": "How can I deploy models on dedicated infrastructure?",
        "relevant_ids": [
            "Deployment & Inference — Inference Endpoints (dedicated)"
        ],
    },
    {
        "question": "What is Text Generation Inference used for?",
        "relevant_ids": [
            "Deployment & Inference — Text Generation Inference"
        ],
    },
    {
        "question": "Which library provides embeddings and reranking?",
        "relevant_ids": ["Core ML Libraries — Sentence Transformers"],
    },
    {
        "question": "Which library provides state-of-the-art vision models?",
        "relevant_ids": ["Core ML Libraries — timm"],
    },
    {
        "question": "What is the capital of France?",
        "relevant_ids": [],
    },
    {
        "question": "How do I bake a chocolate cake?",
        "relevant_ids": [],
    },
    {
        "question": "Who won the FIFA World Cup in 1998?",
        "relevant_ids": [],
    },
]


def cosine_scores(document_vectors, query_vector):
    document_vectors = np.asarray(document_vectors)
    query_vector = np.asarray(query_vector)

    numerator = document_vectors @ query_vector
    denominator = (
        np.linalg.norm(document_vectors, axis=1)
        * np.linalg.norm(query_vector)
    )

    return numerator / np.maximum(denominator, 1e-12)


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source document not found: {SOURCE_PATH}")

    text = SOURCE_PATH.read_text(encoding="utf-8")
    chunks = create_chunks_v2(text)

    print(f"Loaded {len(chunks)} focused chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    document_vectors = embeddings.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    recall_scores = []
    reciprocal_ranks = []
    hit_at_1_scores = []
    out_of_scope_total = 0
    out_of_scope_passes = 0

    for case in TEST_CASES:
        question = case["question"]
        relevant_ids = set(case["relevant_ids"])

        query_vector = embeddings.embed_query(question)
        scores = cosine_scores(document_vectors, query_vector)

        ranked_indices = np.argsort(scores)[::-1][:TOP_K]

        results = [
            {
                "id": chunks[int(i)]["id"],
                "score": float(scores[i]),
            }
            for i in ranked_indices
            if scores[i] >= SCORE_THRESHOLD
        ]

        retrieved_ids = [result["id"] for result in results]

        print(f"\nQuestion: {question}")
        for result in results:
            print(f"  {result['id']} | score={result['score']:.3f}")

        if not relevant_ids:
            out_of_scope_total += 1
            passed = len(results) == 0
            out_of_scope_passes += int(passed)
            print(
                "Out-of-scope rejection: "
                + ("PASS" if passed else "FAIL")
            )
            continue

        relevant_hits = relevant_ids.intersection(retrieved_ids)
        recall = len(relevant_hits) / len(relevant_ids)

        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                reciprocal_rank = 1 / rank
                break

        hit_at_1 = float(
            bool(retrieved_ids) and retrieved_ids[0] in relevant_ids
        )

        recall_scores.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        hit_at_1_scores.append(hit_at_1)

        print(f"Recall@{TOP_K}: {recall:.2f}")
        print(f"Reciprocal rank: {reciprocal_rank:.2f}")
        print(f"Hit@1: {hit_at_1:.0f}")
        print(f"Retrieval: {'PASS' if relevant_hits else 'FAIL'}")

    print("\n--- Chunking V2 Evaluation Summary ---")

    print(f"In-scope questions: {len(recall_scores)}")

    if recall_scores:
        print(
            f"Mean Recall@{TOP_K}: "
            f"{sum(recall_scores) / len(recall_scores):.3f}"
        )
        print(
            f"MRR: "
            f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
        )
        print(
            f"Hit Rate@1: "
            f"{sum(hit_at_1_scores) / len(hit_at_1_scores):.3f}"
        )

    print(
        f"Out-of-scope rejection: "
        f"{out_of_scope_passes}/{out_of_scope_total}"
    )


if __name__ == "__main__":
    main()
