"""
用 sentence-transformers (bge-large-en-v1.5) 为 paper_index.json 中的论文生成 embeddings。
用法: cd backend && uv run python ../scripts/build_embeddings.py

输出: data/embeddings.npy (N × 1024 float32)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 路径
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
DATA_DIR = BACKEND_DIR / "data"
INDEX_PATH = DATA_DIR / "paper_index.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"


def use_direct_huggingface_endpoint() -> None:
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"


def main():
    use_direct_huggingface_endpoint()

    if not INDEX_PATH.exists():
        print(f"ERROR: {INDEX_PATH} not found. Run build_paper_index.py first.")
        sys.exit(1)

    print(f"Loading paper index from {INDEX_PATH}...")
    with open(INDEX_PATH, "r") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")

    # 过滤掉有 error 的论文
    valid_papers = [p for p in papers if "error" not in p]
    print(f"Valid papers: {len(valid_papers)}")

    # 构建 embedding 文本: title + abstract
    texts = []
    for paper in valid_papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text = f"{title} {abstract}".strip()
        if not text:
            text = title or "unknown"
        texts.append(text)

    print(f"Loading sentence-transformers model (BAAI/bge-large-en-v1.5)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print("Model loaded. Encoding texts...")

    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=64)
    elapsed = time.time() - start
    print(f"Encoding done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} papers/sec)")

    # 保存 embeddings
    np.save(str(EMBEDDINGS_PATH), embeddings)
    print(f"Saved embeddings: {embeddings.shape} → {EMBEDDINGS_PATH}")
    print(f"File size: {EMBEDDINGS_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    # 验证
    loaded = np.load(str(EMBEDDINGS_PATH))
    assert loaded.shape == embeddings.shape, f"Shape mismatch: {loaded.shape} vs {embeddings.shape}"
    print(f"Verification OK: {loaded.shape}")


if __name__ == "__main__":
    main()
