
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = PROJECT_ROOT / ".model_cache"


@lru_cache(maxsize=1)
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="onnx/model.onnx",
        cache_dir=str(MODEL_CACHE),
    )

    tokenizer_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="tokenizer.json",
        cache_dir=str(MODEL_CACHE),
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding()

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    return tokenizer, session


def embed_texts(texts):
    if not texts:
        return []

    tokenizer, session = load_model()
    encoded = tokenizer.encode_batch(texts)

    input_ids = np.asarray(
        [item.ids for item in encoded], dtype=np.int64
    )

    attention_mask = np.asarray(
        [item.attention_mask for item in encoded], dtype=np.int64
    )

    token_type_ids = np.asarray(
        [item.type_ids for item in encoded], dtype=np.int64
    )

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if "token_type_ids" in {
        item.name for item in session.get_inputs()
    }:
        inputs["token_type_ids"] = token_type_ids

    token_embeddings = session.run(None, inputs)[0]

    # Attention-mask-aware mean pooling
    mask = attention_mask[:, :, None].astype(np.float32)

    pooled = (token_embeddings * mask).sum(axis=1)
    pooled /= np.maximum(mask.sum(axis=1), 1e-9)

    # L2 normalization
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    normalized = pooled / np.maximum(norms, 1e-12)

    return normalized.tolist()
