"""Print hallucinated answers next to the paragraph the question came from.

Usage:
  python show_hallucinations.py                 # first 5, from baseline_k5
  python show_hallucinations.py outputs/baseline_k5.jsonl 10
"""
import json
import sys
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    in_file = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/baseline_k5.jsonl")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    corpus = {d["id"]: d for d in read_jsonl("data/corpus.jsonl")}
    gold_doc = {q["id"]: q["gold_doc_id"] for q in read_jsonl("data/questions.jsonl")}

    rows = [
        r for r in read_jsonl(in_file)
        if not r["is_answerable"] and r["prediction"].lower().strip(" .") != "unanswerable"
    ]
    print(f"{len(rows)} hallucinated answers in {in_file}. Showing {min(n, len(rows))}:\n")

    for i, r in enumerate(rows[:n], 1):
        doc = corpus[gold_doc[r["id"]]]
        print(f"=== {i}. {r['question']}")
        print(f"model answered: {r['prediction']}")
        print(f"source paragraph ({doc['title']}):\n{doc['text']}\n")


if __name__ == "__main__":
    main()
