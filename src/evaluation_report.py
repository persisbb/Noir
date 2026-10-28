
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation_results.json"
REPORT_FILE = PROJECT_ROOT / "reports" / "evaluation_report.md"


def build_report(results):
    retrieval = results["retrieval"]
    answers = results["answer_evaluation"]
    groundedness = results["llm_groundedness"]
    evidence = results["evidence_checks"]

    lines = [
        "# RAG Evaluation Report",
        "",
        f"**Project:** {results['project']}",
        "",
        "**Report status:** Historical evaluation snapshot",
        "",
        "## 1. Retrieval evaluation",
        "",
        f"- Questions evaluated: {retrieval['questions_total']}",
        f"- In-scope questions: {retrieval['in_scope_questions']}",
        f"- Out-of-scope questions: {retrieval['out_of_scope_questions']}",
        f"- Mean Recall@3: {retrieval['mean_recall_at_3']:.3f}",
        f"- Mean Reciprocal Rank (MRR): {retrieval['mrr']:.3f}",
        f"- Hit Rate@1: {retrieval['hit_rate_at_1']:.3f}",
        (
            "- Out-of-scope rejection: "
            f"{retrieval['out_of_scope_rejections_passed']}/"
            f"{retrieval['out_of_scope_rejections_total']}"
        ),
        "",
        "## 2. Answer evaluation",
        "",
        f"- Passed: {answers['passed']}/{answers['total']}",
        "- Checks performed:",
    ]

    lines.extend(f"  - {check}" for check in answers["checks"])

    lines.extend([
        "",
        "## 3. LLM groundedness evaluation",
        "",
        f"- Passed: {groundedness['passed']}/{groundedness['total']}",
        f"- Method: {groundedness['method']}",
        "",
        "## 4. Retrieval evidence checks",
        "",
        f"- Passed: {evidence['passed']}/{evidence['total']}",
        f"- Method: {evidence['method']}",
        "",
        "## 5. Limitations",
        "",
    ])

    lines.extend(f"- {item}" for item in results["limitations"])

    lines.extend([
        "",
        "## 6. Reproducibility and safety",
        "",
        "- This report uses previously recorded results.",
        "- Generating this report does not call the Gemini API.",
        "- Generating this report does not query or modify Pinecone.",
        "- Re-run the individual evaluators to obtain fresh results.",
        "",
    ])

    return "\n".join(lines)


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}"
        )

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)

    report = build_report(results)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print("Evaluation report created successfully.")
    print(f"Report: {REPORT_FILE}")
    print("\n--- Report preview ---\n")
    print(report)


if __name__ == "__main__":
    main()
