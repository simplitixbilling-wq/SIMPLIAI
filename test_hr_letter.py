"""
Quick end-to-end HR letter generation test.
Tests both PDF and DOCX output formats using the Audit_HR_Letter_Generator_Test template.
Run: python test_hr_letter.py
"""
import json
import os
import sys

# Force UTF-8 stdout so Unicode chars don't crash on Windows cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from bridge import Bridge

def read_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def main():
    print("=" * 60)
    print("BANK RECO AUDIT TEST")
    print("=" * 60)

    b = Bridge()

    # Load template
    tpl_path = os.path.join(BASE, 'instruction_templates.json')
    with open(tpl_path, 'r', encoding='utf-8') as f:
        templates = json.load(f)

    tpl_name = 'Audit_Bank_Reco_Test'
    if tpl_name not in templates:
        print(f"[ERROR] Template '{tpl_name}' not found in instruction_templates.json")
        print(f"Available templates: {list(templates.keys())[:10]}")
        return

    tpl = json.loads(templates[tpl_name])
    print(f"[OK] Template loaded: {tpl_name}")
    print(f"     Role: {tpl.get('role', 'N/A')}")
    print(f"     Format: {tpl.get('format', 'N/A')}")

    # Build instructions from template
    task = tpl.get('task', '')
    steps = tpl.get('steps', '')
    instructions = '\n'.join([x for x in [task, steps] if x]).strip()
    base_format = tpl.get('format', 'csv_json')
    json_schema = tpl.get('json_schema', '')

    # Load test files — audit bank reco inputs
    letters_dir = os.path.join(BASE, 'tests', 'ui_agent_inputs', 'audit_reco')
    file_names = [
        'bank_statement_apr_2025_sample.csv',
        'cash_book_apr_2025_sample.csv',
    ]
    files = []
    for fn in file_names:
        path = os.path.join(letters_dir, fn)
        content = read_file(path)
        files.append({
            'name': fn,
            'size': len(content.encode('utf-8')),
            'content': content,
        })
        print(f"[OK] Loaded: {fn} ({len(content)} chars)")

    # Check model status
    model_info = b.get_model_status()
    print(f"\n[MODEL] Status: {model_info}")

    if not model_info.get('loaded'):
        print("\n[INFO] No model loaded. Discovering available models...")
        models = b.get_models()
        if not models:
            print("[ERROR] No models found. Cannot run test.")
            return
        print(f"[INFO] Available models: {[m['label'] for m in models]}")
        chosen = models[0]
        print(f"[INFO] Selecting: {chosen['label']}")
        b.select_model(chosen['label'])
        b.load_model()
        print("[INFO] Waiting for model to load (up to 120s)...")
        import time
        for i in range(60):
            time.sleep(2)
            status = b.get_model_status()
            if status.get('loaded'):
                print(f"[OK] Model loaded: {status}")
                model_info = status
                break
            if i % 5 == 0:
                print(f"[INFO] Still loading... ({(i + 1) * 2}s)")
        else:
            print("[ERROR] Model did not load within 120s. Aborting.")
            return

    # Run csv_json test
    print("\n" + "=" * 60)
    print("TEST 1: csv_json bank reco generation")
    print("=" * 60)
    result = b.process_files_with_ai(files, instructions, output_format='csv_json', json_schema=json_schema)
    _check_result(result, 'CSV_JSON')

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def _check_result(result, fmt):
    if not result:
        print(f"[FAIL] {fmt}: process_files_with_ai returned None/empty")
        return

    success = result.get('success', False)
    error = result.get('error')
    file_path = result.get('file_path')
    warning = result.get('warning')
    ai_text = result.get('response_text', '')

    if error:
        print(f"[FAIL] {fmt} error: {error}")
        return

    if not success:
        print(f"[FAIL] {fmt}: success=False, result={result}")
        return

    print(f"[OK] {fmt} generation succeeded")

    if warning:
        print(f"[WARN] {fmt}: {warning}")

    if file_path:
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        print(f"[FILE] {fmt}: {file_path}")
        print(f"       Exists: {exists}, Size: {size:,} bytes")
        if not exists or size == 0:
            print(f"[FAIL] {fmt}: output file is missing or empty!")
            return
        if fmt == 'PDF':
            _check_pdf(file_path)
        elif fmt == 'DOCX':
            _check_docx(file_path)
    elif ai_text:
        print(f"[OK] {fmt}: inline response, {len(ai_text)} chars")
        _print_preview(ai_text)
    else:
        print(f"[FAIL] {fmt}: no file_path and no response_text in result")


def _check_pdf(path):
    try:
        import fitz
        doc = fitz.open(path)
        pages = len(doc)
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        doc.close()
        print(f"[PDF] Pages: {pages}, Total text: {len(all_text)} chars")
        if len(all_text) < 100:
            print("[FAIL] PDF text content is too short (< 100 chars) — likely blank!")
        else:
            print("[OK] PDF has sufficient text content")
            _print_preview(all_text)
    except Exception as e:
        print(f"[WARN] Could not inspect PDF content: {e}")


def _check_docx(path):
    try:
        from docx import Document
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        all_text = '\n'.join(paragraphs)
        print(f"[DOCX] Paragraphs: {len(paragraphs)}, Total text: {len(all_text)} chars")
        if len(all_text) < 100:
            print("[FAIL] DOCX text content is too short (< 100 chars) — likely blank!")
        else:
            print("[OK] DOCX has sufficient text content")
            _print_preview(all_text)
    except Exception as e:
        print(f"[WARN] Could not inspect DOCX content: {e}")


def _print_preview(text, chars=500):
    preview = text[:chars].replace('\n', ' | ')
    print(f"[PREVIEW] {preview}")


if __name__ == '__main__':
    main()
