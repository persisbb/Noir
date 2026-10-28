
from retriever import search


TEST_CASES = [
    {
        "question": "What is the Hugging Face Hub?",
        "relevant_ids": ["doc-1"],
    },
    {
        "question": "Where can I host models, datasets, and Spaces?",
        "relevant_ids": ["doc-1"],
    },
    {
        "question": "What does the Hub Python Library do?",
        "relevant_ids": ["doc-1"],
    },
    {
        "question": "Which tools interact with Hugging Face services?",
        "relevant_ids": ["doc-1"],
    },
    {
        "question": "What are Inference Endpoints?",
        "relevant_ids": ["doc-2"],
    },
    {
        "question": "How can I deploy models on dedicated infrastructure?",
        "relevant_ids": ["doc-2"],
    },
    {
        "question": "What is Text Generation Inference used for?",
        "relevant_ids": ["doc-2"],
    },
    {
        "question": "Which library provides embeddings and reranking?",
        "relevant_ids": ["doc-3"],
    },
    {
        "question": "Which library provides state-of-the-art vision models?",
        "relevant_ids": ["doc-3"],
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


def main():
    top_k = 3
    recall_scores = []
    reciprocal_ranks = []
    hit_at_1_scores = []
    out_of_scope_passes = 0
    out_of_scope_total = 0

    for case in TEST_CASES:
        question = case["question"]
        relevant_ids = set(case["relevant_ids"])

        results = search(question, top_k=top_k)
        matches = results["matches"]
        retrieved_ids = [
            str(match["id"]) for match in matches
        ]

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print(f"Expected relevant IDs: {sorted(relevant_ids)}")
        print(f"Retrieved IDs: {retrieved_ids}")

        print("Retrieved documents and scores:")
        for match in matches:
            print(
                f"  {match['id']} | "
                f"score={match.get('score', 0):.3f}"
            )

        # Out-of-scope questions should have no relevant matches.
        if not relevant_ids:
            out_of_scope_total += 1
            passed = len(matches) == 0
            out_of_scope_passes += int(passed)

            print(
                f"Out-of-scope rejection: "
                f"{'PASS' if passed else 'FAIL'}"
            )
            continue

        # Check whether relevant documents were retrieved.
        hits = relevant_ids.intersection(retrieved_ids)

        recall_at_k = len(hits) / len(relevant_ids)
        recall_scores.append(recall_at_k)

        hit_at_1 = (
            1.0
            if retrieved_ids and retrieved_ids[0] in relevant_ids
            else 0.0
        )
        hit_at_1_scores.append(hit_at_1)

        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

        print(f"Recall@{top_k}: {recall_at_k:.2f}")
        print(f"Reciprocal rank: {reciprocal_rank:.2f}")
        print(f"Retrieval: {'PASS' if hits else 'FAIL'}")

    print("\n" + "=" * 60)
    print("--- Evaluation Summary ---")

    if recall_scores:
        mean_recall = sum(recall_scores) / len(recall_scores)
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        hit_rate_at_1 = sum(hit_at_1_scores) / len(hit_at_1_scores)

        print(f"In-scope questions: {len(recall_scores)}")
        print(f"Mean Recall@{top_k}: {mean_recall:.3f}")
        print(f"MRR: {mrr:.3f}")
        print(f"Hit Rate@1: {hit_rate_at_1:.3f}")
    else:
        print("No in-scope questions were evaluated.")

    print(
        f"Out-of-scope rejection: "
        f"{out_of_scope_passes}/{out_of_scope_total}"
    )


if __name__ == "__main__":
    main()
