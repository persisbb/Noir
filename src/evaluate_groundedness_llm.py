
import json

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

from retriever import search
from app import generate_answer


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

JUDGE_MODEL = "gemini-3.5-flash-lite"

TEST_CASES = [
    {
        "question": "What is the Hugging Face Hub?",
        "expect_abstention": False,
    },
    {
        "question": "Which library provides embeddings and reranking?",
        "expect_abstention": False,
    },
    {
        "question": "What is the capital of France?",
        "expect_abstention": True,
    },
]


def evaluate_groundedness(question, answer, context):
    prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) answer.

Determine whether the answer's factual claims are supported by the
provided context.

Return only valid JSON with these fields:
{{
  "grounded": true,
  "explanation": "Short explanation"
}}

Use grounded=true only if the answer is supported by the context.
If the answer contains unsupported factual claims, use grounded=false.
An explicit statement that the documents lack enough information is
acceptable when the context does not answer the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""

    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    return json.loads(response.text)


def main():
    passed = 0

    for case in TEST_CASES:
        question = case["question"]
        expect_abstention = case["expect_abstention"]

        results = search(question, top_k=3)
        context = "\n\n".join(
            match["metadata"].get("text", "")
            for match in results["matches"]
        )

        try:
            answer = generate_answer(question)
        except Exception as error:
            print(f"\nQuestion: {question}")
            print(f"API error: {error}")
            print("Test: ERROR (answer generation did not complete)")
            continue

        if expect_abstention and not results["matches"]:
            # No relevant context was retrieved, so the app should abstain.
            groundedness = {
                "grounded": (
                    "I don't have enough information in the provided documents."
                    in answer
                ),
                "explanation": (
                    "Checked whether the app abstained when retrieval returned "
                    "no sufficiently relevant chunks."
                ),
            }
        else:
            groundedness = evaluate_groundedness(
                question,
                answer,
                context,
            )

        success = groundedness["grounded"]

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        print(f"Grounded: {groundedness['grounded']}")
        print(f"Explanation: {groundedness['explanation']}")
        print(f"Test: {'PASS' if success else 'FAIL'}")

        if success:
            passed += 1

    print(f"\nSummary: {passed}/{len(TEST_CASES)} groundedness tests passed")


if __name__ == "__main__":
    main()
