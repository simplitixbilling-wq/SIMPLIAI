"""Plugin system — scans plugins/ folder for .py files with register(app) hooks."""

import os
from typing import Dict, List


# ══════════════════════════════════════════════════════════════════════════
# #39  Plugin System
# ══════════════════════════════════════════════════════════════════════════
class PluginManager:
    """Scans plugins/ folder for .py files with a register(app) entry point."""

    def __init__(self, plugins_dir: str):
        self.plugins_dir = plugins_dir
        self.loaded: Dict[str, dict] = {}  # {name: {"module": mod, "info": str}}
        os.makedirs(plugins_dir, exist_ok=True)

    def discover(self) -> List[str]:
        """Return list of plugin filenames (without .py)."""
        if not os.path.isdir(self.plugins_dir):
            return []
        return [
            f[:-3] for f in sorted(os.listdir(self.plugins_dir))
            if f.endswith(".py") and not f.startswith("_")
        ]

    def load_all(self, app):
        """Import every plugin and call register(app)."""
        import importlib.util
        for name in self.discover():
            if name in self.loaded:
                continue
            path = os.path.join(self.plugins_dir, name + ".py")
            try:
                spec = importlib.util.spec_from_file_location(f"plugins.{name}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "register"):
                    mod.register(app)
                info = getattr(mod, "PLUGIN_INFO", name)
                self.loaded[name] = {"module": mod, "info": info}
            except Exception as e:
                self.loaded[name] = {"module": None, "info": f"⚠ {e}"}

    def unload(self, name: str, app):
        """Call unregister(app) if available, then remove."""
        entry = self.loaded.pop(name, None)
        if entry and entry["module"] and hasattr(entry["module"], "unregister"):
            try:
                entry["module"].unregister(app)
            except Exception:
                pass

    def list_plugins(self) -> List[dict]:
        """Return [{name, info, loaded}] for UI display."""
        discovered = set(self.discover())
        result = []
        for n in discovered:
            loaded = n in self.loaded and self.loaded[n]["module"] is not None
            info = self.loaded.get(n, {}).get("info", n)
            result.append({"name": n, "info": info, "loaded": loaded})
        return result
