"""Step 4: score model answers with exact match (EM) and token-level precision / recall / F1.

Usage:
  python metrics.py outputs/baseline_k5.jsonl

Output:
  outputs/<name>_scored.jsonl   every row plus its em / precision / recall / f1
  prints a summary table and examples where EM and F1 disagree
"""
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

REFUSAL = "unanswerable"


# ---------- normalization (same rules as the official SQuAD script) ----------

def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_refusal(prediction):
    return normalize(prediction) == REFUSAL


# ---------- metrics for one prediction vs one gold answer ----------

def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def token_prf(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)  # overlap, counting repeats
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)  # how much of the answer is correct
    recall = overlap / len(gold_tokens)     # how much of the gold answer was found
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ---------- score one row ----------

def score_row(row):
    pred = row["prediction"]

    if not row["is_answerable"]:
        # Correct output is a refusal; anything else is a hallucination
        ok = float(is_refusal(pred))
        return {"em": ok, "precision": ok, "recall": ok, "f1": ok, "hallucinated": not ok}

    if is_refusal(pred):
        return {"em": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "hallucinated": False}

    # Several gold answers may be acceptable: take the best match
    em = max(exact_match(pred, g) for g in row["gold_answers"])
    p, r, f1 = max((token_prf(pred, g) for g in row["gold_answers"]), key=lambda x: x[2])
    return {"em": em, "precision": p, "recall": r, "f1": f1, "hallucinated": False}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def main():
    in_file = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/baseline_k5.jsonl")
    with open(in_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    for row in rows:
        row.update(score_row(row))

    out_file = in_file.with_name(in_file.stem + "_scored.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ans = [r for r in rows if r["is_answerable"]]
    unans = [r for r in rows if not r["is_answerable"]]
    ans_hit = [r for r in ans if r["gold_in_context"]]
    ans_miss = [r for r in ans if not r["gold_in_context"]]

    def line(name, group):
        print(f"{name:<34} {len(group):>4} {mean(r['em'] for r in group):>6.3f} "
              f"{mean(r['f1'] for r in group):>6.3f} {mean(r['precision'] for r in group):>6.3f} "
              f"{mean(r['recall'] for r in group):>6.3f}")

    print(f"\nscored -> {out_file}\n")
    print(f"{'group':<34} {'n':>4} {'EM':>6} {'F1':>6} {'P':>6} {'R':>6}")
    print("-" * 66)
    line("answerable (all)", ans)
    line("  gold paragraph retrieved", ans_hit)
    line("  gold paragraph NOT retrieved", ans_miss)
    line("overall (SQuAD v2 style)", rows)

    print(f"\nunanswerable: correctly refused {mean(not r['hallucinated'] for r in unans):.1%}, "
          f"hallucinated {mean(r['hallucinated'] for r in unans):.1%}")
    print(f"answerable:   refused {mean(is_refusal(r['prediction']) for r in ans):.1%}")

    # Where EM says "wrong" but F1 says "mostly right": shows why EM is too strict
    partial = [r for r in ans if r["em"] == 0 and r["f1"] >= 0.5]
    print(f"\nEM = 0 but F1 >= 0.5: {len(partial)} answerable questions. Examples:")
    for r in partial[:5]:
        print(f"  gold: {r['gold_answers']}  |  model: {r['prediction']}  |  F1={r['f1']:.2f}")


if __name__ == "__main__":
    main()
