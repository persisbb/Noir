# NOIR — AI-Powered Document Intelligence

**A deployed Retrieval-Augmented Generation (RAG) application for querying documents using semantic search and generative AI.**

[Live Demo](https://noir-vca7.onrender.com) | [GitHub Repository](https://github.com/persisbb/Noir)

## Overview

NOIR is an end-to-end document intelligence application that allows users to upload PDF documents and ask natural-language questions about their contents.

The system uses semantic vector search to retrieve relevant document passages and Google's Gemini model to generate context-grounded answers with source citations.

Built as an AI engineering portfolio project, NOIR demonstrates the integration of document processing, vector databases, lightweight inference, backend development, and cloud deployment.

## Features

- PDF upload and text extraction
- Document chunking and vector embeddings
- Semantic retrieval using Pinecone
- Context-grounded answer generation with Gemini
- Document and page-level source citations
- User authentication and protected routes
- Responsive Django web interface
- Cloud deployment using Render

## Architecture

```text
User
 |
 v
Django Web Application
 |
 +--> PDF Upload
 |      |
 |      v
 |   Text Extraction (pypdf)
 |      |
 |      v
 |   Recursive Text Chunking
 |      |
 |      v
 |   ONNX Embeddings (384 dimensions)
 |      |
 |      v
 |   Pinecone Vector Database
 |
 +--> User Question
        |
        v
     Query Embedding
        |
        v
     Pinecone Semantic Search
        |
        v
     Retrieved Context
        |
        v
     Gemini LLM
        |
        v
     Answer + Source Citations
```

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, Django |
| Language Model | Google Gemini |
| Embeddings | all-MiniLM-L6-v2 |
| Inference | ONNX Runtime |
| Vector Database | Pinecone |
| PDF Processing | pypdf |
| Text Splitting | LangChain Text Splitters |
| Authentication | Django Authentication |
| Database | PostgreSQL (Neon) |
| Deployment | Render, Gunicorn, WhiteNoise |

## Engineering Decisions

### Lightweight ONNX Inference

NOIR uses ONNX Runtime instead of loading the full PyTorch and Sentence Transformers stack in production.

This reduces embedding inference overhead and makes deployment more practical on memory-constrained infrastructure.

### Memory-Aware Document Processing

Document chunks are embedded in small batches to control peak inference memory usage.

### Retrieval-Augmented Generation

Instead of relying solely on an LLM's pretrained knowledge, NOIR retrieves relevant passages from