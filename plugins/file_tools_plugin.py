"""Plugin: system & file management tools.

Commands:
  /ls [path]          — List files and folders
  /tree [path]        — Show directory tree (2 levels deep)
  /sort <path> [ext]  — Sort files into subfolders by extension
  /find <path> <name> — Find files matching a name pattern
  /size [path]        — Show disk usage / folder size
  /mv <src> <dst>     — Move or rename a file
  /cp <src> <dst>     — Copy a file or folder
  /mkdir <path>       — Create a directory
  /info <file>        — Show file metadata (size, dates, type)
"""

import os
import shutil
import time
import glob
import stat
from datetime import datetime

PLUGIN_INFO = "File & system tools: /ls /tree /sort /find /size /mv /cp /mkdir /info"

# Restrict operations to user-accessible paths (block system dirs)
_BLOCKED_ROOTS = {"C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)", "/usr", "/bin", "/sbin", "/boot", "/etc"}


def _safe_path(raw: str) -> str:
    """Resolve and validate path. Returns absolute path or raises ValueError."""
    p = os.path.abspath(os.path.expanduser(raw.strip()))
    for blocked in _BLOCKED_ROOTS:
        if p.lower().startswith(blocked.lower()):
            raise ValueError(f"Access denied: {blocked} is a protected system directory")
    return p


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _file_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# ── Command handlers ─────────────────────────────────────────

def _cmd_ls(app, text: str):
    parts = text.split(None, 1)
    path = _safe_path(parts[1] if len(parts) > 1 else ".")
    if not os.path.isdir(path):
        return {"content": f"Not a directory: {path}"}
    entries = sorted(os.listdir(path))
    if not entries:
        return {"content": f"📂 {path}\n(empty)"}
    lines = [f"📂 **{path}**  ({len(entries)} items)\n"]
    for name in entries:
        fp = os.path.join(path, name)
        if os.path.isdir(fp):
            lines.append(f"  📁 {name}/")
        else:
            sz = _human_size(os.path.getsize(fp))
            lines.append(f"  📄 {name}  ({sz})")
    return {"content": "\n".join(lines)}


def _cmd_tree(app, text: str):
    parts = text.split(None, 1)
    path = _safe_path(parts[1] if len(parts) > 1 else ".")
    if not os.path.isdir(path):
        return {"content": f"Not a directory: {path}"}

    lines = [f"📂 **{path}**\n"]
    max_depth = 2
    max_items = 200
    count = 0

    def _walk(p, prefix, depth):
        nonlocal count
        if depth > max_depth or count > max_items:
            return
        try:
            entries = sorted(os.listdir(p))
        except PermissionError:
            lines.append(f"{prefix}⛔ (access denied)")
            return
        dirs = [e for e in entries if os.path.isdir(os.path.join(p, e))]
        files = [e for e in entries if not os.path.isdir(os.path.join(p, e))]
        for d in dirs:
            count += 1
            if count > max_items:
                lines.append(f"{prefix}... (truncated)")
                return
            lines.append(f"{prefix}📁 {d}/")
            _walk(os.path.join(p, d), prefix + "  ", depth + 1)
        for f in files:
            count += 1
            if count > max_items:
                lines.append(f"{prefix}... (truncated)")
                return
            lines.append(f"{prefix}📄 {f}")

    _walk(path, "  ", 0)
    return {"content": "\n".join(lines)}


def _cmd_sort(app, text: str):
    """Sort files in a directory into subfolders by extension."""
    parts = text.split()
    if len(parts) < 2:
        return {"content": "Usage: /sort <path> [ext1,ext2,...]\nExample: /sort ~/Downloads\nExample: /sort ~/Downloads pdf,docx"}
    path = _safe_path(parts[1])
    if not os.path.isdir(path):
        return {"content": f"Not a directory: {path}"}

    filter_exts = None
    if len(parts) > 2:
        filter_exts = {e.strip().lower().lstrip('.') for e in parts[2].split(',')}

    moved = {}
    errors = []
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        if not os.path.isfile(fp):
            continue
        ext = os.path.splitext(name)[1].lower().lstrip('.')
        if not ext:
            ext = "_no_extension"
        if filter_exts and ext not in filter_exts:
            continue
        dest_dir = os.path.join(path, ext.upper() + "_files")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest):
            errors.append(f"Skipped (exists): {name}")
            continue
        try:
            shutil.move(fp, dest)
            moved.setdefault(ext, []).append(name)
        except Exception as e:
            errors.append(f"Failed {name}: {e}")

    lines = [f"📂 Sorted files in **{path}**\n"]
    total = sum(len(v) for v in moved.values())
    for ext, files in sorted(moved.items()):
        lines.append(f"  📁 {ext.upper()}_files/ → {len(files)} file(s)")
    if errors:
        lines.append(f"\n⚠ {len(errors)} error(s):")
        for e in errors[:5]:
            lines.append(f"  - {e}")
    if not total and not errors:
        lines.append("No files to sort.")
    else:
        lines.append(f"\n✅ Moved {total} file(s)")
    return {"content": "\n".join(lines)}


def _cmd_find(app, text: str):
    parts = text.split(None, 2)
    if len(parts) < 3:
        return {"content": "Usage: /find <path> <pattern>\nExample: /find ~/Documents *.pdf"}
    path = _safe_path(parts[1])
    pattern = parts[2].strip()
    if not os.path.isdir(path):
        return {"content": f"Not a directory: {path}"}

    search = os.path.join(path, "**", pattern)
    matches = glob.glob(search, recursive=True)
    if not matches:
        return {"content": f"No files matching `{pattern}` in {path}"}

    lines = [f"🔍 Found **{len(matches)}** match(es) for `{pattern}`\n"]
    for m in matches[:50]:
        rel = os.path.relpath(m, path)
        sz = _human_size(os.path.getsize(m)) if os.path.isfile(m) else "dir"
        lines.append(f"  {rel}  ({sz})")
    if len(matches) > 50:
        lines.append(f"  ... and {len(matches) - 50} more")
    return {"content": "\n".join(lines)}


def _cmd_size(app, text: str):
    parts = text.split(None, 1)
    path = _safe_path(parts[1] if len(parts) > 1 else ".")
    if os.path.isfile(path):
        s = os.stat(path)
        return {"content": f"📄 {os.path.basename(path)}: {_human_size(s.st_size)}"}
    if not os.path.isdir(path):
        return {"content": f"Path not found: {path}"}

    total = 0
    file_count = 0
    dir_count = 0
    for root, dirs, files in os.walk(path):
        dir_count += len(dirs)
        for f in files:
            file_count += 1
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass

    # Also show disk usage
    disk = shutil.disk_usage(path)
    lines = [
        f"📂 **{path}**",
        f"  Folder size: {_human_size(total)}",
        f"  Files: {file_count}  |  Subdirs: {dir_count}",
        f"",
        f"💾 Disk: {_human_size(disk.used)} / {_human_size(disk.total)} used  ({_human_size(disk.free)} free)",
    ]
    return {"content": "\n".join(lines)}


def _cmd_mv(app, text: str):
    parts = text.split(None, 2)
    if len(parts) < 3:
        return {"content": "Usage: /mv <source> <destination>"}
    src = _safe_path(parts[1])
    dst = _safe_path(parts[2])
    if not os.path.exists(src):
        return {"content": f"Source not found: {src}"}
    shutil.move(src, dst)
    return {"content": f"✅ Moved:\n  {src}\n  → {dst}"}


def _cmd_cp(app, text: str):
    parts = text.split(None, 2)
    if len(parts) < 3:
        return {"content": "Usage: /cp <source> <destination>"}
    src = _safe_path(parts[1])
    dst = _safe_path(parts[2])
    if not os.path.exists(src):
        return {"content": f"Source not found: {src}"}
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return {"content": f"✅ Copied:\n  {src}\n  → {dst}"}


def _cmd_mkdir(app, text: str):
    parts = text.split(None, 1)
    if len(parts) < 2:
        return {"content": "Usage: /mkdir <path>"}
    path = _safe_path(parts[1])
    os.makedirs(path, exist_ok=True)
    return {"content": f"✅ Created directory: {path}"}


def _cmd_info(app, text: str):
    parts = text.split(None, 1)
    if len(parts) < 2:
        return {"content": "Usage: /info <file_or_folder>"}
    path = _safe_path(parts[1])
    if not os.path.exists(path):
        return {"content": f"Not found: {path}"}
    s = os.stat(path)
    is_dir = os.path.isdir(path)
    lines = [
        f"{'📁' if is_dir else '📄'} **{os.path.basename(path)}**",
        f"  Path: {path}",
        f"  Type: {'Directory' if is_dir else 'File'}",
        f"  Size: {_human_size(s.st_size)}",
        f"  Created:  {_file_time(s.st_ctime)}",
        f"  Modified: {_file_time(s.st_mtime)}",
        f"  Accessed: {_file_time(s.st_atime)}",
    ]
    if not is_dir:
        ext = os.path.splitext(path)[1]
        lines.append(f"  Extension: {ext or '(none)'}")
        lines.append(f"  Read-only: {not os.access(path, os.W_OK)}")
    return {"content": "\n".join(lines)}


# ── Registration ─────────────────────────────────────────────

_COMMANDS = {
    "/ls":    (_cmd_ls,    "List files and folders"),
    "/tree":  (_cmd_tree,  "Show directory tree"),
    "/sort":  (_cmd_sort,  "Sort files into subfolders by extension"),
    "/find":  (_cmd_find,  "Find files by name pattern"),
    "/size":  (_cmd_size,  "Show folder/disk size"),
    "/mv":    (_cmd_mv,    "Move or rename a file"),
    "/cp":    (_cmd_cp,    "Copy a file or folder"),
    "/mkdir": (_cmd_mkdir, "Create a directory"),
    "/info":  (_cmd_info,  "Show file metadata"),
}


def register(app):
    for cmd, (handler, desc) in _COMMANDS.items():
        app.register_plugin_command(cmd, handler, plugin_name="file_tools_plugin", description=desc)
    print(f"[PLUGIN] file_tools loaded — {len(_COMMANDS)} commands")


def unregister(app):
    for cmd in _COMMANDS:
        app.unregister_plugin_command(cmd)
