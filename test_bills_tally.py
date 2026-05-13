"""
Bills-to-Tally conversion test.
Template: Bills_To_Tally_Test
Data:     tests/ui_agent_inputs/bills_tally/
Run:      python test_bills_tally.py
"""
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from bridge import Bridge


def read_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def _check_result(result, label):
    print(f"\n[RESULT] {label}")
    if not result:
        print("  [FAIL] No result returned.")
        return
    if isinstance(result, dict) and result.get('error'):
        print(f"  [FAIL] Error: {result['error']}")
        return

    text = ''
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict):
        text = result.get('content') or result.get('text') or str(result)

    print(f"  Chars  : {len(text)}")
    print(f"  Preview: {text[:300]}")

    # Check for JSON rows array
    if '"rows"' in text or "'rows'" in text:
        print("  [OK] 'rows' key found in output")
    else:
        print("  [WARN] 'rows' key NOT found — may be non-JSON output")

    # Check for Tally ledger columns
    for col in ['Date', 'Voucher Type', 'Party Ledger', 'Taxable Amount', 'Gross Amount']:
        if col in text:
            print(f"  [OK] Column present: {col}")
        else:
            print(f"  [WARN] Column missing: {col}")

    # Check for output file path
    if isinstance(result, dict):
        out = result.get('output_file') or result.get('file_path') or result.get('path')
        if out:
            print(f"  Output file: {out}")
            if os.path.exists(out):
                size = os.path.getsize(out)
                print(f"  [OK] File exists: {size} bytes")
            else:
                print(f"  [WARN] File path returned but file not found.")


def main():
    print("=" * 60)
    print("BILLS TO TALLY CONVERSION TEST")
    print("=" * 60)

    b = Bridge()

    # Load template
    tpl_path = os.path.join(BASE, 'instruction_templates.json')
    with open(tpl_path, 'r', encoding='utf-8') as f:
        templates = json.load(f)

    tpl_name = 'Bills_To_Tally_Test'
    if tpl_name not in templates:
        print(f"[ERROR] Template '{tpl_name}' not found.")
        print(f"Available: {list(templates.keys())}")
        return

    tpl = json.loads(templates[tpl_name])
    print(f"[OK] Template loaded: {tpl_name}")
    print(f"     Role   : {tpl.get('role', 'N/A')}")
    print(f"     Format : {tpl.get('format', 'N/A')}")

    task = tpl.get('task', '')
    steps = tpl.get('steps', '')
    instructions = '\n'.join([x for x in [task, steps] if x]).strip()
    base_format = tpl.get('format', 'csv_json')
    json_schema = tpl.get('json_schema', '')

    # Load test files
    data_dir = os.path.join(BASE, 'tests', 'ui_agent_inputs', 'bills_tally')
    file_names = [
        'vendor_bills_sample.csv',
        'tally_mapping_guide.txt',
    ]
    files = []
    for fn in file_names:
        path = os.path.join(data_dir, fn)
        content = read_file(path)
        files.append({
            'name': fn,
            'size': len(content.encode('utf-8')),
            'content': content,
        })
        print(f"[OK] Loaded: {fn} ({len(content)} chars)")

    # Check / load model
    model_info = b.get_model_status()
    print(f"\n[MODEL] Status: {model_info}")

    if not model_info.get('loaded'):
        print("\n[INFO] No model loaded. Discovering available models...")
        models = b.get_models()
        if not models:
            print("[ERROR] No models found.")
            return
        print(f"[INFO] Available models: {[m['label'] for m in models]}")
        target = next((m for m in models if 'gemma' in m['label'].lower() or
                       '(a)' in m['label'].lower()), models[0])
        print(f"[INFO] Selecting: {target['label']}")
        b.load_model(target['path'])

        print("[INFO] Waiting for model to load (up to 120s)...")
        import time
        for _ in range(24):
            time.sleep(5)
            status = b.get_model_status()
            if status.get('loaded'):
                break
        if not b.get_model_status().get('loaded'):
            print("[ERROR] Model did not load in time.")
            return

    print(f"[OK] Model loaded: {b.get_model_status()}")

    # Run csv_json test
    print("\n" + "=" * 60)
    print("TEST 1: csv_json Bills-to-Tally conversion")
    print("=" * 60)
    result = b.process_files_with_ai(
        files, instructions, output_format='csv_json', json_schema=json_schema
    )
    _check_result(result, 'CSV_JSON')

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
