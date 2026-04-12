import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


def read_api_key(explicit_key: str | None) -> str:
    if explicit_key:
        return explicit_key.strip()

    env_key = os.environ.get("SIMPLE_AI_API_KEY", "").strip()
    if env_key:
        return env_key

    key_file = Path("local_api_key.txt")
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        "API key not found. Set SIMPLE_AI_API_KEY or create local_api_key.txt in workspace root."
    )


def extension_to_language(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".html": "html",
        ".css": "css",
        ".md": "markdown",
        ".java": "java",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
        ".php": "php",
        ".rb": "ruby",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".xml": "xml",
        ".sql": "sql",
    }
    return mapping.get(ext, "text")


def extract_code_block(text: str) -> str:
    text = text.strip()
    block = re.search(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)```", text)
    if block:
        return block.group(1).strip("\n")
    return text


def build_prompt(file_path: Path, content: str, issue: str, language: str) -> str:
    return (
        "You are a senior software engineer fixing code issues.\n"
        "Task:\n"
        f"- File: {file_path.name}\n"
        f"- Language: {language}\n"
        f"- Fix request: {issue}\n\n"
        "Rules:\n"
        "- Preserve behavior unless the fix requires change.\n"
        "- Keep the same file structure.\n"
        "- Do not add explanations, only return the full corrected file content.\n"
        "- Return plain code only (no markdown fences).\n\n"
        "Current file content:\n"
        f"{content}"
    )


def call_local_api(api_url: str, api_key: str, prompt: str) -> str:
    url = api_url.rstrip("/") + "/v1/chat"
    payload = {
        "text": prompt,
        "role": "You are a precise code-fixing assistant.",
        "task": "Fix code issues and return corrected full file content.",
        "steps": "1) Identify defects 2) Fix them 3) Return final complete code only",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-API-Key", api_key)

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to call local API: {e}") from e

    try:
        obj = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid API response: {raw[:300]}") from e

    if obj.get("error"):
        raise RuntimeError(obj["error"])

    text = str(obj.get("text", "")).strip()
    if not text:
        raise RuntimeError("Empty response from AI")
    return extract_code_block(text)


def write_output(source_file: Path, fixed_code: str, apply: bool, output_path: str | None) -> Path:
    if output_path:
        out = Path(output_path)
        out.write_text(fixed_code, encoding="utf-8")
        return out

    if apply:
        backup = source_file.with_suffix(source_file.suffix + ".bak")
        shutil.copyfile(source_file, backup)
        source_file.write_text(fixed_code, encoding="utf-8")
        return source_file

    out = source_file.with_suffix(source_file.suffix + ".fixed")
    out.write_text(fixed_code, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Send current file to SIMPLE_AI local API for code fixes.")
    parser.add_argument("--file", required=True, help="Path to source file")
    parser.add_argument(
        "--issue",
        default="Fix bugs and code issues in this file while preserving behavior.",
        help="Issue description for the AI",
    )
    parser.add_argument("--apply", action="store_true", help="Overwrite original file (creates .bak)")
    parser.add_argument("--output", help="Write fixed code to explicit output file")
    parser.add_argument("--api-url", default="http://127.0.0.1:8765", help="Local API base URL")
    parser.add_argument("--api-key", help="Optional API key (otherwise env/file is used)")
    args = parser.parse_args()

    src = Path(args.file)
    if not src.exists():
        print(f"ERROR: File not found: {src}")
        return 1

    if src.stat().st_size > 800_000:
        print("ERROR: File too large (>800KB). Split the file or target smaller sections.")
        return 1

    try:
        code = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        code = src.read_text(encoding="utf-8", errors="replace")

    language = extension_to_language(src)

    try:
        key = read_api_key(args.api_key)
        prompt = build_prompt(src, code, args.issue, language)
        fixed_code = call_local_api(args.api_url, key, prompt)
        out = write_output(src, fixed_code, apply=args.apply, output_path=args.output)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    if args.apply:
        print(f"Updated file: {out}")
    else:
        print(f"Fixed output written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
