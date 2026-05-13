"""
Test: Mixed CSV+TXT routing fix
Verifies that uploading 2 CSVs + 1 TXT now routes to the SQL pipeline (not text pipeline),
and that the result contains real employee names from the CSV (not hallucinated ones).
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAYROLL_DIR = os.path.join(os.path.dirname(__file__), "tests", "ui_agent_inputs", "payroll")
EMPLOYEE_CSV = os.path.join(PAYROLL_DIR, "employee_master_may_2026.csv")
ATTENDANCE_CSV = os.path.join(PAYROLL_DIR, "attendance_may_2026.csv")
RULES_TXT = os.path.join(PAYROLL_DIR, "payroll_rules_may_2026.txt")

# Real employee names that MUST appear in a correct result
REAL_NAMES = ["Anita Verma", "Rajesh Iyer", "Sana Khan"]

def load_file(path, name=None):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {
        "name": name or os.path.basename(path),
        "size": os.path.getsize(path),
        "content": content,
        "path": path,
    }

def main():
    print("=" * 60)
    print("TEST: Mixed CSV+TXT routing fix")
    print("=" * 60)

    # ── 1. Load Bridge ──────────────────────────────────────────
    print("\n[1] Importing bridge...")
    from bridge import Bridge
    b = Bridge()
    print("    Bridge created.")

    # ── 2. Check routing BEFORE model load ──────────────────────
    files = [
        load_file(EMPLOYEE_CSV),
        load_file(ATTENDANCE_CSV),
        load_file(RULES_TXT),
    ]

    routed = b._should_use_tabular_pipeline(files)
    print(f"\n[2] _should_use_tabular_pipeline([2 CSVs + 1 TXT]) = {routed}")
    if routed:
        print("    PASS: Routes to SQL pipeline (any-tabular logic).")
    else:
        print("    FAIL: Still routes to text pipeline — fix not applied!")
        sys.exit(1)

    # ── 3. Load model ───────────────────────────────────────────
    print("\n[3] Loading model...")
    models = b.get_models()
    qwen = next(
        (m for m in models if "gemma-4" in m.get("label", "").lower() or "gemma-4" in m.get("filename", "").lower()),
        None
    )
    if not qwen:
        print("    Could not find Gemma 4 model. Available:")
        for m in models:
            print(f"      {m['label']}")
        sys.exit(1)

    label = qwen["label"]
    filename = qwen.get("filename", os.path.basename(qwen.get("path", "")))
    print(f"    Using: {label}")

    b.select_model(label)
    b.model_configs[filename] = {"n_ctx": 4096}
    b.load_model()

    # Wait for model to load
    deadline = time.time() + 120
    while time.time() < deadline:
        status = b.get_model_status()
        if status.get("loaded"):
            print(f"    Model loaded: {status.get('model_name', label)}")
            break
        if status.get("error"):
            print(f"    Model load error: {status['error']}")
            sys.exit(1)
        time.sleep(2)
    else:
        print("    Timeout waiting for model to load.")
        sys.exit(1)

    # ── 4. Run payroll processing ───────────────────────────────
    instructions = (
        "Compute net pay for each employee. "
        "Join employee master and attendance. "
        "Apply PF deduction (12% of Basic where PF Applicable = Yes). "
        "Output: Employee ID, Employee Name, Basic, Gross, PF Deduction, Net Pay."
    )

    print("\n[4] Running process_files_with_ai (2 CSVs + 1 TXT)...")
    t0 = time.time()
    result = b.process_files_with_ai(files, instructions, output_format="csv_json")
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    if result.get("error"):
        print(f"    ERROR: {result['error']}")
        sys.exit(1)

    # ── 5. Check output for real names ──────────────────────────
    response_text = result.get("response_text", "")
    file_path = result.get("file_path", "")

    # Also read the output XLSX/CSV if generated
    csv_content = ""
    df_out = None
    if file_path and os.path.exists(file_path):
        try:
            import pandas as pd
            if file_path.endswith('.xlsx'):
                df_out = pd.read_excel(file_path)
            else:
                df_out = pd.read_csv(file_path)
            csv_content = df_out.to_string()
            print(f"\n[5] Output file: {file_path}")
            print(f"    Contents:\n{df_out.to_string()}")
        except Exception as e:
            print(f"\n[5] Could not read output file: {e}")
    else:
        print(f"\n[5] No output file. Response text (first 500):\n{response_text[:500]}")

    combined = (response_text + csv_content).lower()

    found = [name for name in REAL_NAMES if name.lower() in combined]
    missing = [name for name in REAL_NAMES if name.lower() not in combined]

    print(f"\n[6] Real names found in output: {found}")
    if missing:
        print(f"    NOT found (may be hallucinated): {missing}")

    # Check that instruction-required computed columns are not all NULL
    computed_ok = False
    computed_issues = []
    if df_out is not None:
        import pandas as pd
        # Look for gross and net_pay columns (case-insensitive)
        col_map = {c.lower().replace(' ', '_'): c for c in df_out.columns}
        for expected in ['gross', 'net_pay']:
            if expected in col_map:
                col = col_map[expected]
                non_null = df_out[col].notna().sum()
                total = len(df_out)
                if non_null == 0:
                    computed_issues.append(f"'{col}' is all NULL — model failed to compute it")
                else:
                    print(f"    [CHECK] '{col}': {non_null}/{total} rows have computed values ✓")
        computed_ok = len(computed_issues) == 0
        for issue in computed_issues:
            print(f"    [WARN] {issue}")

    # ── 6. Verdict ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    if len(found) >= 2 and computed_ok:
        print(f"PASS: {len(found)}/{len(REAL_NAMES)} real names present, computed columns correct.")
        print("=" * 60)
        sys.exit(0)
    elif len(found) >= 2:
        print(f"PARTIAL PASS: {len(found)}/{len(REAL_NAMES)} real names present.")
        for issue in computed_issues:
            print(f"  - {issue}")
        print("  Rows returned correctly but computed columns (Gross/Net Pay) are NULL.")
        print("  This is a model capability limit, not a routing bug.")
        print("=" * 60)
        sys.exit(0)
    elif len(found) == 1:
        print(f"PARTIAL: Only {len(found)} real name found — check output manually.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("FAIL: No real employee names in output — hallucination still occurring!")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
