import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.rag_manager import RAGManager


def normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def main() -> None:
    docs_path = r"C:\Users\Chandana\Downloads\docsforbgv"
    rag_name = "bgv_docsforbgv_eval_curated"
    eval_file = Path(__file__).with_name("rag_eval_docsforbgv_curated.json")

    with eval_file.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    rm = RAGManager(base_directory="rag_databases")

    if rag_name not in rm.list_databases():
        rm.create_from_folder(docs_path, rag_name)

    hits = 0
    details = []

    for case in cases:
        expected = case["expected"]
        query = case["query"]

        chunks = rm.retrieve(rag_name, query, k=5)
        top_text = "\n".join(c for c, _ in chunks)

        ok = normalize(expected) in normalize(top_text)
        hits += int(ok)
        details.append(
            {
                "label": case["label"],
                "query": query,
                "expected": expected,
                "hit": ok,
                "top_score": chunks[0][1] if chunks else 0.0,
            }
        )

    total = len(cases)
    acc = (hits / total) if total else 0.0

    print("CURATED_RAG_EVAL")
    print(f"RAG: {rag_name}")
    print(f"cases={total} hits={hits} accuracy={acc:.3f}")
    for d in details:
        print(f"- {d['label']}: {'HIT' if d['hit'] else 'MISS'} | expected={d['expected']} | top_score={d['top_score']:.4f}")


if __name__ == "__main__":
    main()
