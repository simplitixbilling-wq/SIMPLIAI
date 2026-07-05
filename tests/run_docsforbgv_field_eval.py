import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.rag_manager import RAGManager


def norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def collect_cases(db, max_cases: int = 60):
    by_source = defaultdict(list)
    for idx, chunk in enumerate(db.chunks):
        src = db.metadata[idx].get("source", "") if idx < len(db.metadata) else ""
        if src:
            by_source[src].append(chunk)

    patterns = [
        ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "What is the PAN in {source}?"),
        ("passport", re.compile(r"\b[A-Z][0-9]{7}\b"), "What is the passport number in {source}?"),
        ("cin", re.compile(r"\b[A-Z0-9]{10,}\b"), "What is the CIN in {source}?"),
        ("ref_no", re.compile(r"\b[A-Z0-9-]{5,}\b"), "What is the reference number in {source}?"),
        ("date", re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"), "What is a date in {source}?"),
        ("amount", re.compile(r"\b\d[\d,]*\.\d{2}\b"), "What is an amount in {source}?"),
    ]

    cases = []
    for source, chunks in by_source.items():
        text = "\n".join(chunks)
        for label, rx, template in patterns:
            m = rx.search(text)
            if not m:
                continue
            expected = m.group(0)
            if label == "cin" and not re.match(r"^[A-Z]{1,4}[0-9A-Z]{8,}$", expected):
                continue
            query = template.format(source=source)
            query_generic = query.replace(f" in {source}", "")
            cases.append(
                {
                    "source": source,
                    "label": label,
                    "query": query,
                    "query_generic": query_generic,
                    "expected": expected,
                }
            )
            if len(cases) >= max_cases:
                return cases

    return cases


def evaluate(rm, rag_name: str, cases, constrained: bool, include_source_in_query: bool):
    hits = 0
    for case in cases:
        query = case["query"] if include_source_in_query else case["query_generic"]
        kwargs = {"source_filter": [case["source"]]} if constrained else {}
        chunks = rm.retrieve(rag_name, query, k=5, **kwargs)
        top_text = "\n".join(c for c, _ in chunks)
        if norm(case["expected"]) in norm(top_text):
            hits += 1
    total = len(cases)
    return hits, total, (hits / total if total else 0.0)


def main():
    docs_path = r"C:\Users\Chandana\Downloads\docsforbgv"
    rag_name = "bgv_docsforbgv_eval_fields"

    rm = RAGManager(base_directory="rag_databases")
    if rag_name not in rm.list_databases():
        if not os.path.isdir(docs_path):
            print("docs folder missing", docs_path)
            return
        rm.create_from_folder(docs_path, rag_name)

    db = rm.databases[rag_name]
    cases = collect_cases(db)

    base_hits, total, base_acc = evaluate(rm, rag_name, cases, constrained=False, include_source_in_query=True)
    generic_hits, _, generic_acc = evaluate(rm, rag_name, cases, constrained=False, include_source_in_query=False)
    constrained_hits, _, constrained_acc = evaluate(rm, rag_name, cases, constrained=True, include_source_in_query=False)

    print("DOCSFORBGV_FIELD_EVAL")
    print(f"cases={total}")
    print(f"baseline_with_source_in_query hits={base_hits} accuracy={base_acc:.3f}")
    print(f"generic_query_no_source hits={generic_hits} accuracy={generic_acc:.3f}")
    print(f"generic_query_with_source_filter hits={constrained_hits} accuracy={constrained_acc:.3f}")


if __name__ == "__main__":
    main()
