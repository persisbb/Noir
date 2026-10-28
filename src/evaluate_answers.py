
from app import generate_answer


FALLBACK = (
    "I don't have enough information in the provided documents."
)

TEST_CASES = [
    {
        "question": "What is the Hugging Face Hub?",
        "expected_keywords": ["models", "datasets"],
        "expect_abstention": False,
        "expect_source": True,
    },
    {
        "question": "Which library provides embeddings and reranking?",
        "expected_keywords": ["Sentence Transformers"],
        "expect_abstention": False,
        "expect_source": True,
    },
    {
        "question": "What are Inference Endpoints?",
        "expected_keywords": ["dedicated", "infrastructure"],
        "expect_abstention": False,
        "expect_source": True,
    },
    {
        "question": "What is the capital of France?",
        "expected_keywords": [],
        "expect_abstention": True,
        "expect_source": False,
    },

    {
        "question": "How do I bake a chocolate cake?",
        "expected_keywords": [],
        "expect_abstention": True,
        "expect_source": False,
    },
    {
        "question": "Who won the FIFA World Cup in 1998?",
        "expected_keywords": [],
        "expect_abstention": True,
        "expect_source": False,
    },

]


def main():
    passed = 0

    for case in TEST_CASES:
        question = case["question"]
        expected = case["expected_keywords"]
        expect_abstention = case["expect_abstention"]
        expect_source = case["expect_source"]

        try:
            answer = generate_answer(question)
        except Exception as error:
            print(f"\nQuestion: {question}")
            print(f"API or execution error: {error}")
            print("Test: ERROR (not counted as PASS or FAIL)")
            continue
        answer_lower = answer.lower()

        # Separate the generated answer from the source line.
        answer_text = answer.split("\n\nSource:", 1)[0].strip()

        checks = {}

        if expect_abstention:
            checks["abstention"] = FALLBACK.lower() in answer_text.lower()
        else:
            checks["keywords"] = all(
                keyword.lower() in answer_text.lower()
                for keyword in expected
            )

        checks["source"] = (
            "source:" in answer_lower
            if expect_source
            else "source:" not in answer_lower
        )


        success = all(checks.values())

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        print(f"Checks: {checks}")
        print(f"Test: {'PASS' if success else 'FAIL'}")

        if success:
            passed += 1

    print(f"\nSummary: {passed}/{len(TEST_CASES)} answer tests passed")


if __name__ == "__main__":
    main()
