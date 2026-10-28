
import sys
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.http import Http404
import json
from pathlib import Path
from django.conf import settings

import logging
import tempfile

from django.views.decorators.http import require_http_methods
from .forms import PDFUploadForm


# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import the existing RAG answer-generation function.
from src.pdf_pinecone import upload_pdf
from src.pdf_qa import answer_pdf_question
from src.app import generate_answer
from src.retriever import (
    get_index_stats,
    list_source_chunks,
    get_source_chunk,
)


def home(request):
    """Render the NOIR chat interface."""
    return render(request, "chat/index.html")


@require_POST
def ask(request):
    """Receive a question and return the RAG answer with its sources."""
    question = request.POST.get("question", "").strip()

    if not question:
        return JsonResponse(
            {"error": "Please enter a question."},
            status=400,
        )

    try:
        result = generate_answer(question)

        # Separate the answer from the source labels.
        answer, separator, source_text = result.partition(
            "\n\nSource: "
        )

        sources = []

        if separator and source_text.strip():
            sources = [
                source.strip()
                for source in source_text.split(",")
                if source.strip()
            ]

        return JsonResponse(
            {
                "answer": answer.strip(),
                "sources": sources,
            }
        )

    except Exception:
        # Keep technical details out of the browser response.
        return JsonResponse(
            {"error": "Something went wrong. Please try again."},
            status=500,
        )




def sources(request):
    """Display live Pinecone statistics in the Sources page."""

    stats = None
    error = None

    try:
        stats = get_index_stats()
    except Exception as exc:
        print(f"Unable to fetch Pinecone statistics: {exc}")
        error = "Live index statistics are temporarily unavailable."

    return render(
        request,
        "chat/source.html",
        {
            "stats": stats,
            "stats_error": error,
        },
    )



def source_explorer(request):
    """Display the chunks stored in Pinecone."""
    chunks = []
    error = None

    try:
        chunks = list_source_chunks()
    except Exception as exc:
        print(f"Unable to list source chunks: {exc}")
        error = "Unable to load source chunks right now."

    return render(
        request,
        "chat/explorer.html",
        {"chunks": chunks, "error": error},
    )


def source_detail(request, vector_id):
    """Display the text and metadata of one stored chunk."""
    try:
        chunk = get_source_chunk(vector_id)
    except Exception as exc:
        print(f"Unable to retrieve chunk: {exc}")
        raise Http404("Chunk could not be retrieved.")

    if chunk is None:
        raise Http404("Chunk not found.")

    return render(
        request,
        "chat/chunk_detail.html",
        {"chunk": chunk},
    )



def evaluations(request):
    results_path = (
        Path(settings.BASE_DIR).parent
        / "data"
        / "evaluation_results.json"
    )

    try:
        with results_path.open("r", encoding="utf-8") as file:
            results = json.load(file)
        error = None
    except (OSError, json.JSONDecodeError) as exc:
        results = None
        error = f"Unable to load evaluation results: {exc}"

    return render(
        request,
        "chat/evaluations.html",
        {
            "results": results,
            "error": error,
        },
    )



logger = logging.getLogger(__name__)


@require_http_methods(["GET", "POST"])
def documents(request):
    """Upload and index a PDF in the local NOIR workspace."""

    form = PDFUploadForm()
    uploaded_document = request.session.get("uploaded_document")
    error = None

    if request.method == "POST":
        form = PDFUploadForm(
            request.POST,
            request.FILES,
        )

        if form.is_valid():
            pdf_file = form.cleaned_data["document"]

            try:
                # The temporary file is removed automatically.
                with tempfile.TemporaryDirectory() as temp_dir:
                    from django.utils.text import get_valid_filename

                    safe_filename = get_valid_filename(Path(pdf_file.name).name)
                    pdf_path = Path(temp_dir) / safe_filename

                    with pdf_path.open("wb") as destination:
                        for chunk in pdf_file.chunks():
                            destination.write(chunk)

                    # Reuse the tested PDF-to-Pinecone pipeline.
                    result = upload_pdf(str(pdf_path))

                uploaded_document = result
                request.session["uploaded_document"] = result

            except Exception:
                logger.exception("PDF indexing failed")
                error = (
                    "We couldn't process this PDF. "
                    "Please try another document."
                )

    return render(
        request,
        "chat/documents.html",
        {
            "form": form,
            "uploaded_document": uploaded_document,
            "error": error,
        },
    )



@require_POST
def ask_document(request):
    """Answer a question using one uploaded PDF."""

    question = request.POST.get("question", "").strip()
    document_id = request.POST.get("document_id", "").strip()

    if not question or not document_id:
        return JsonResponse(
            {"error": "Question and document ID are required."},
            status=400,
        )

    try:
        result = answer_pdf_question(
            question=question,
            document_id=document_id,
        )

        return JsonResponse(result)

    except Exception:
        logger.exception("PDF question answering failed")

        return JsonResponse(
            {
                "error": (
                    "Unable to answer this question right now. "
                    "Please try again."
                )
            },
            status=500,
        )
