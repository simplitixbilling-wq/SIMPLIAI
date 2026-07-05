"""Shared utility functions used across all modules."""

import os
import sys


def resource_path(relative_path: str) -> str:
    """Path to bundled read-only resources inside the PyInstaller package."""
    base_path = getattr(sys, "_MEIPASS", None)
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def app_data_path(relative_path: str = "") -> str:
    """Path for writable user data (models, RAG databases, saved chats).
    Always resolves to a folder next to the .exe (or script during dev)."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path) if relative_path else base
