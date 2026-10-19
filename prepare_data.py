"""Step 1: build the question set and document corpus from SQuAD v2 dev.

Outputs:
  data/corpus.jsonl     every unique paragraph in SQuAD v2 dev (the retrieval corpus)
  data/questions.jsonl  500 sampled questions, half answerable / half unanswerable
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_QUESTIONS = 500
OUT = Path("data")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    # Corpus: every unique paragraph becomes one document
    ctx_to_id = {}
    corpus = []
    for row in ds:
        ctx = row["context"]
        if ctx not in ctx_to_id:
            ctx_to_id[ctx] = f"doc_{len(corpus)}"
            corpus.append({"id": ctx_to_id[ctx], "title": row["title"], "text": ctx})

    # Questions: balanced sample. Unanswerable questions are the main
    # hallucination probe: the context looks relevant but has no answer.
    rng = random.Random(SEED)
    answerable = [r for r in ds if r["answers"]["text"]]
    unanswerable = [r for r in ds if not r["answers"]["text"]]
    half = N_QUESTIONS // 2
    sample = rng.sample(answerable, half) + rng.sample(unanswerable, N_QUESTIONS - half)
    rng.shuffle(sample)

    questions = [
        {
            "id": r["id"],
            "question": r["question"],
            "answers": sorted(set(r["answers"]["text"])),
            "is_answerable": bool(r["answers"]["text"]),
            "gold_doc_id": ctx_to_id[r["context"]],
        }
        for r in sample
    ]

    write_jsonl(OUT / "corpus.jsonl", corpus)
    write_jsonl(OUT / "questions.jsonl", questions)

    n_ans = sum(q["is_answerable"] for q in questions)
    print(f"corpus: {len(corpus)} docs")
    print(f"questions: {len(questions)} ({n_ans} answerable, {len(questions) - n_ans} unanswerable)")
    print("example:", json.dumps(questions[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
