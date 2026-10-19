"""Step 3: answer each question using its top-k retrieved paragraphs.

Usage:
  python generate.py          # default k=5
  python generate.py --k 1

Output:
  outputs/baseline_k{k}.jsonl   one row per question with the model's answer
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from llm import chat

DATA = Path("data")
OUT = Path("outputs")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context passages.\n"
    "Rules:\n"
    "- Answer with the shortest exact phrase from the context that answers the question.\n"
    "- Do not explain or write a full sentence.\n"
    "- If the context does not contain the answer, reply exactly: unanswerable"
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_messages(question, passages):
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    questions = read_jsonl(DATA / "questions.jsonl")
    corpus = {d["id"]: d["text"] for d in read_jsonl(DATA / "corpus.jsonl")}
    retrieval = {r["id"]: r["retrieved"] for r in read_jsonl(DATA / "retrieval.jsonl")}

    def answer_one(q):
        doc_ids = retrieval[q["id"]][: args.k]
        prediction = chat(build_messages(q["question"], [corpus[d] for d in doc_ids]))
        return {
            "id": q["id"],
            "question": q["question"],
            "gold_answers": q["answers"],
            "is_answerable": q["is_answerable"],
            "gold_in_context": q["gold_doc_id"] in doc_ids,
            "context_doc_ids": doc_ids,
            "prediction": prediction,
        }

    # 8 requests in parallel: ~1 min instead of ~6 for 500 sequential calls
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(tqdm(pool.map(answer_one, questions), total=len(questions), desc=f"answering k={args.k}"))

    OUT.mkdir(exist_ok=True)
    out_file = OUT / f"baseline_k{args.k}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick sanity check (real metrics come in Step 4)
    def said_unanswerable(r):
        return r["prediction"].lower().strip(" .") == "unanswerable"

    ans = [r for r in rows if r["is_answerable"]]
    unans = [r for r in rows if not r["is_answerable"]]
    print(f"\nsaved {out_file}")
    print(f"answerable questions   -> said 'unanswerable': {sum(map(said_unanswerable, ans))}/{len(ans)}")
    print(f"unanswerable questions -> said 'unanswerable': {sum(map(said_unanswerable, unans))}/{len(unans)}")
    print("\n3 examples:")
    for r in rows[:3]:
        print(f"  Q: {r['question']}\n     gold: {r['gold_answers'] or ['(no answer)']}  |  model: {r['prediction']}")


if __name__ == "__main__":
    main()
