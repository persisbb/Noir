
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = PROJECT_ROOT / ".model_cache"

MODEL_CACHE.mkdir(parents=True, exist_ok=True)

for filename in ["onnx/model.onnx", "tokenizer.json"]:
    path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=filename,
        cache_dir=str(MODEL_CACHE),
    )
    print(f"Ready: {filename}")
    print(f"Location: {path}")

print("ONNX model files ready.")
