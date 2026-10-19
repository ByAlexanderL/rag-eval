"""Step 2: embed corpus + questions with OpenAI embeddings, retrieve top-k by cosine similarity.

Outputs:
  data/retrieval.jsonl   top-k doc ids + similarity scores for each question
  prints recall@k (did the gold paragraph show up in the top k?)
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 10
BATCH = 100
DATA = Path("data")
CACHE = Path("cache")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_texts(client, texts, name):
    """Embed texts in batches. Results are cached on disk, keyed by a hash of
    the model + inputs, so re-running costs nothing unless the data changes."""
    key = hashlib.sha256((EMBED_MODEL + "\n".join(texts)).encode()).hexdigest()[:12]
    cache_file = CACHE / f"emb_{name}_{key}.npy"
    if cache_file.exists():
        print(f"[cache] {name}: loaded {cache_file.name}")
        return np.load(cache_file)

    vecs = []
    for i in tqdm(range(0, len(texts), BATCH), desc=f"embedding {name}"):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + BATCH])
        vecs.extend(d.embedding for d in resp.data)

    arr = np.array(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)  # unit length -> dot product = cosine
    np.save(cache_file, arr)
    return arr


def recall_at_k(questions, results, k, subset=None):
    pairs = [(q, r) for q, r in zip(questions, results) if subset is None or subset(q)]
    hits = sum(q["gold_doc_id"] in r["retrieved"][:k] for q, r in pairs)
    return hits / len(pairs)


def main():
    load_dotenv()
    client = OpenAI()
    CACHE.mkdir(exist_ok=True)

    corpus = read_jsonl(DATA / "corpus.jsonl")
    questions = read_jsonl(DATA / "questions.jsonl")

    # Prepend the article title: it gives the embedding useful topic context
    doc_vecs = embed_texts(client, [f"{d['title']}\n{d['text']}" for d in corpus], "corpus")
    q_vecs = embed_texts(client, [q["question"] for q in questions], "questions")

    scores = q_vecs @ doc_vecs.T  # shape (n_questions, n_docs)
    top_idx = np.argsort(-scores, axis=1)[:, :TOP_K]

    results = [
        {
            "id": q["id"],
            "retrieved": [corpus[i]["id"] for i in idx],
            "scores": [round(float(row[i]), 4) for i in idx],
        }
        for q, idx, row in zip(questions, top_idx, scores)
    ]
    write_jsonl(DATA / "retrieval.jsonl", results)

    print(f"\n{'k':>4} | {'recall (all)':>12} | {'recall (answerable)':>19}")
    for k in (1, 3, 5, 10):
        all_r = recall_at_k(questions, results, k)
        ans_r = recall_at_k(questions, results, k, subset=lambda q: q["is_answerable"])
        print(f"{k:>4} | {all_r:>12.3f} | {ans_r:>19.3f}")


if __name__ == "__main__":
    main()
