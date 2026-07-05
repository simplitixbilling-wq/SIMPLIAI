"""Security helpers for local-only integrations."""

from __future__ import annotations

import hmac
import ast
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse


LOCAL_API_MAX_BODY_BYTES = 5 * 1024 * 1024


def constant_time_equals(provided: str, expected: str) -> bool:
    provided = str(provided or "")
    expected = str(expected or "")
    return bool(provided and expected and hmac.compare_digest(provided, expected))


def is_loopback_host(host: str) -> bool:
    raw = str(host or "").strip().strip("[]").lower()
    if raw == "localhost":
        return True
    try:
        return ipaddress.ip_address(raw).is_loopback
    except ValueError:
        return False


def is_allowed_local_origin(origin: str) -> bool:
    if not origin:
        return True
    parsed = urlparse(str(origin))
    if parsed.scheme == "file":
        return True
    if parsed.scheme not in {"http", "https"}:
        return False
    return is_loopback_host(parsed.hostname or "")


def parse_content_length(value: str, max_bytes: int = LOCAL_API_MAX_BODY_BYTES) -> tuple[int | None, str | None]:
    try:
        length = int(value or 0)
    except (TypeError, ValueError):
        return None, "Invalid Content-Length"
    if length < 0:
        return None, "Invalid Content-Length"
    if length > max_bytes:
        return None, f"Request body too large; limit is {max_bytes} bytes"
    return length, None


def safe_local_path(path: str, *, allowed_roots: list[str], must_exist: bool = False) -> str:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("Path is required")
    if raw.startswith("\\\\"):
        raise ValueError("UNC/network paths are not allowed")

    resolved = Path(raw).expanduser().resolve()
    if must_exist and not resolved.exists():
        raise ValueError("Path does not exist")

    root_paths = [Path(root).expanduser().resolve() for root in allowed_roots if str(root or "").strip()]
    if not root_paths:
        raise ValueError("No allowed roots configured")

    resolved_norm = os.path.normcase(str(resolved))
    for root in root_paths:
        root_norm = os.path.normcase(str(root))
        if resolved_norm == root_norm or resolved_norm.startswith(root_norm + os.sep):
            return str(resolved)
    raise ValueError("Path is outside allowed local roots")


def validate_restricted_python_snippet(code: str, allowed_imports: list[str]) -> str:
    """Return an error string when plugin snippet code violates the local sandbox policy."""
    allowed = {str(name).strip().split(".", 1)[0] for name in allowed_imports if str(name).strip()}
    try:
        tree = ast.parse(str(code or ""), filename="<plugin_snippet>", mode="exec")
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    banned_calls = {"compile", "eval", "exec", "getattr", "globals", "input", "locals", "open", "setattr"}
    banned_nodes = (ast.ClassDef, ast.Delete, ast.Global, ast.Lambda, ast.Nonlocal, ast.With)

    for node in ast.walk(tree):
        if isinstance(node, banned_nodes):
            return f"Unsafe syntax blocked: {type(node).__name__}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("__"):
            return "Unsafe dunder function blocked"
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in banned_calls:
                return f"Unsafe name blocked: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"Unsafe attribute blocked: {node.attr}"
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id in banned_calls:
                return f"Unsafe call blocked: {target.id}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed:
                    return f"Import blocked: {root}"

    return ""
