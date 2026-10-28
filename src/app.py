from src.llm_service import generate_text, LLMUnavailableError

try:
    from .retriever import search
except ImportError:
    from retriever import search




def generate_answer(question: str):
    # Retrieve relevant documents
    results = search(question, top_k=3)

    # If nothing relevant was retrieved, don't call the LLM
    if not results["matches"]:
        return "I don't have enough information in the provided documents."

    contexts = []
    sources = []

    for match in results["matches"]:
        text = match["metadata"]["text"]
        chunk_id = match["metadata"].get("chunk_id")

        contexts.append(text)
        sources.append(f"huggingface_docs (chunk {chunk_id})")

    context = "\n\n".join(contexts)
    source_text = ", ".join(sources)

    prompt = f"""
    You are a RAG question-answering assistant.

    Use the provided CONTEXT to answer the QUESTION.

    IMPORTANT:
    - Answer the question using the information in the context.
    - Do not say that information is missing if the context contains relevant information.
    - Keep the answer concise and clear.
    - Do not include a source, citation, reference, or "Source:" line in your answer.
    - If the context truly does not contain enough information to answer the question, say:
    "I don't have enough information in the provided documents."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """


    try:
        answer = generate_text(prompt)
    except LLMUnavailableError:
        return "I'm unable to generate an answer right now. Please try again."

    return f"{answer}\n\nSource: {source_text}"


if __name__ == "__main__":
    question = input("Ask a question: ")

    answer = generate_answer(question)

    print("\nAnswer:")
    print(answer)