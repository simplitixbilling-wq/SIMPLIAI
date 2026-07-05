"""
Sequential test suite — 6 templates, small model.
Runs: CS, Income Tax, Payroll, Recruitment, Statutory Audit, Labour Compliance
Run: python test_suite_sequential.py
"""
import json
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from app_core.bridge import Bridge

# ── Test matrix ──────────────────────────────────────────────────────────────
TESTS = [
    {
        'label': 'CS Drafting',
        'template': 'CS_Drafting_Test',
        'data_dir': 'cs',
        'files': ['board_minutes_brief.txt', 'cs_style_guide.txt', 'directors_report_brief.txt'],
    },
    {
        'label': 'Income Tax Computation',
        'template': 'Income_Tax_Computation_Test',
        'data_dir': 'income_tax',
        'files': ['income_case_ay_2026_27.txt', 'deduction_proofs_sample.csv'],
    },
    {
        'label': 'Payroll Calculation',
        'template': 'Payroll_Calculation_Test',
        'data_dir': 'payroll',
        'files': ['employee_master_may_2026.csv', 'attendance_may_2026.csv', 'payroll_rules_may_2026.txt'],
    },
    {
        'label': 'Recruitment Screening',
        'template': 'Recruitment_Screening_Test',
        'data_dir': 'recruitment',
        'files': ['job_description_data_analyst.txt', 'candidate_profiles_sample.csv'],
    },
    {
        'label': 'Statutory Audit Process',
        'template': 'Statutory_Audit_Process_Test',
        'data_dir': 'statutory_audit',
        'files': ['trial_balance_mar_2026.csv', 'prior_year_observations.txt', 'planning_brief.txt'],
    },
    {
        'label': 'Labour Compliance Report',
        'template': 'Labour_Compliance_Report_Test',
        'data_dir': 'labour_compliance',
        'files': ['employee_register_apr_2026.csv', 'wage_and_challan_notes_apr_2026.txt', 'compliance_calendar.txt'],
    },
]


def read_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_small_model(b):
    """Select and load the smallest available model (Qwen/non-gemma)."""
    models = b.get_models()
    if not models:
        print("[ERROR] No models found.")
        return False

    print(f"[INFO] Available models: {[m['label'] for m in models]}")

    # Prefer Chat-agent Qwen → any Qwen → anything not gemma-4 → fallback first
    def priority(m):
        lbl = m['label'].lower()
        if 'chat-agent' in lbl and 'qwen' in lbl:
            return 0
        if 'qwen' in lbl:
            return 1
        if 'gemma-4' not in lbl and '(a)' not in lbl:
            return 2
        return 3

    target = sorted(models, key=priority)[0]
    print(f"[INFO] Selected model: {target['label']}")
    b.select_model(target['label'])

    # Force a safe context window so the small model doesn't OOM trying to allocate 65K tokens
    import pathlib
    filename = pathlib.Path(b.model_path).name
    b.model_configs[filename] = {"n_ctx": 4096}
    print(f"[INFO] Forcing n_ctx=4096 for {filename}")

    # Patch _load_model_thread to surface errors immediately
    _orig_thread = b._load_model_thread.__func__
    def _patched_thread(self_b):
        try:
            _orig_thread(self_b)
        except Exception as exc:
            print(f"[MODEL-ERROR] Thread crashed: {exc}")
    import types
    b._load_model_thread = types.MethodType(_patched_thread, b)

    b.load_model()

    print("[INFO] Waiting for model to load (up to 120s)...")
    for _ in range(24):
        time.sleep(5)
        if b.get_model_status().get('loaded'):
            break

    if not b.get_model_status().get('loaded'):
        print("[ERROR] Model did not load in time.")
        return False

    print(f"[OK] Model ready: {b.get_model_status()}")
    return True


def check_result(result, label, fmt):
    ok = True
    text = ''
    if not result:
        print(f"  [FAIL] No result returned.")
        return False
    if isinstance(result, dict) and result.get('error'):
        print(f"  [FAIL] Error: {result['error']}")
        return False
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict):
        text = result.get('content') or result.get('text') or str(result)

    print(f"  Chars  : {len(text)}")
    print(f"  Preview: {text[:300]}")

    if fmt == 'csv_json':
        if '"rows"' in text or "'rows'" in text:
            print("  [OK] 'rows' key present")
        else:
            print("  [WARN] 'rows' key NOT found")
            ok = False
    elif fmt == 'pdf':
        if len(text) > 100:
            print("  [OK] Non-empty PDF/text output")
        else:
            print("  [WARN] Output very short")
            ok = False

    if isinstance(result, dict):
        out = result.get('output_file') or result.get('file_path') or result.get('path')
        if out:
            exists = os.path.exists(out)
            size = os.path.getsize(out) if exists else 0
            status = f"{size} bytes" if exists else "NOT FOUND"
            print(f"  Output : {out} — {status}")

    return ok


def run_test(b, templates, t):
    tpl_name = t['template']
    label = t['label']

    print(f"\n{'='*60}")
    print(f"TEST: {label}  [{tpl_name}]")
    print('='*60)

    if tpl_name not in templates:
        print(f"[SKIP] Template '{tpl_name}' not found.")
        return 'SKIP'

    tpl = json.loads(templates[tpl_name])
    fmt = tpl.get('format', 'csv_json')
    task = tpl.get('task', '')
    steps = tpl.get('steps', '')
    instructions = '\n'.join([x for x in [task, steps] if x]).strip()
    json_schema = tpl.get('json_schema', '')

    data_dir = os.path.join(BASE, 'tests', 'ui_agent_inputs', t['data_dir'])
    files = []
    for fn in t['files']:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue
        content = read_file(path)
        files.append({'name': fn, 'size': len(content.encode('utf-8')), 'content': content})
        print(f"  [OK] {fn} ({len(content)} chars)")

    if not files:
        print("[FAIL] No files loaded.")
        return 'FAIL'

    result = b.process_files_with_ai(files, instructions, output_format=fmt,
                                     json_schema=json_schema if json_schema else None)
    passed = check_result(result, label, fmt)
    return 'PASS' if passed else 'WARN'


def main():
    print('='*60)
    print('SEQUENTIAL TEST SUITE  (small model)')
    print('='*60)

    b = Bridge()

    # Load templates
    with open(os.path.join(BASE, 'instruction_templates.json'), 'r', encoding='utf-8') as f:
        templates = json.load(f)

    # Load small model
    status = b.get_model_status()
    if status.get('loaded'):
        name = status.get('name', '')
        # If big model already loaded, switch to small
        if 'gemma-4' in name.lower() or '(a)' in name.lower():
            print(f"[INFO] Big model loaded ({name}), switching to small...")
            if not load_small_model(b):
                return
        else:
            print(f"[OK] Small model already loaded: {name}")
    else:
        if not load_small_model(b):
            return

    # Run all tests
    summary = []
    for t in TESTS:
        outcome = run_test(b, templates, t)
        summary.append((t['label'], outcome))

    # Summary
    print(f"\n{'='*60}")
    print('SUMMARY')
    print('='*60)
    for lbl, outcome in summary:
        icon = {'PASS': '[PASS]', 'WARN': '[WARN]', 'FAIL': '[FAIL]', 'SKIP': '[SKIP]'}.get(outcome, outcome)
        print(f"  {icon}  {lbl}")
    print('='*60)


if __name__ == '__main__':
    main()
