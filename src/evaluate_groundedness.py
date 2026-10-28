
from retriever import search


TEST_CASES = [
    {
        "question": "What is the Hugging Face Hub?",
        "expected_evidence": [
            "models",
            "datasets",
            "Spaces",
        ],
    },
    {
        "question": "What is the capital of France?",
        "expected_evidence": [],
    },
]


def evaluate_evidence(question, expected_evidence):
    results = search(question, top_k=3)

    retrieved_text = " ".join(
        match["metadata"].get("text", "")
        for match in results["matches"]
    ).lower()

    if not expected_evidence:
        passed = not results["matches"]
        print(f"Question: {question}")
        print(f"Retrieved chunks: {len(results['matches'])}")
        print(f"Abstention evidence check: {'PASS' if passed else 'FAIL'}")
        return passed

    missing = [
        phrase for phrase in expected_evidence
        if phrase.lower() not in retrieved_text
    ]

    passed = not missing

    print(f"Question: {question}")
    print(f"Retrieved chunks: {len(results['matches'])}")
    print(f"Expected evidence: {expected_evidence}")
    print(f"Missing evidence: {missing}")
    print(f"Evidence check: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    passed = 0

    for case in TEST_CASES:
        print()
        if evaluate_evidence(
            case["question"],
            case["expected_evidence"],
        ):
            passed += 1

    print(f"\nSummary: {passed}/{len(TEST_CASES)} evidence tests passed")


if __name__ == "__main__":
    main()
