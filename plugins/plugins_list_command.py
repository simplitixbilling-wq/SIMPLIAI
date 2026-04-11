"""Sample plugin: adds /plugins command to show loaded plugins."""

import types

PLUGIN_INFO = "Sample: /plugins command"

_prev_send = None
_patched_send = None



def _handle_plugins(app, text: str):
    if str(text or "").strip().lower() != "/plugins":
        return {"handled": False}
    try:
        rows = app.plugin_manager.list_plugins()
    except Exception as exc:
        rows = [{"name": "error", "loaded": False, "info": str(exc)}]

    loaded = [p["name"] for p in rows if p.get("loaded")]
    unloaded = [p["name"] for p in rows if not p.get("loaded")]

    parts = [
        "Plugin status",
        "",
        f"Loaded: {', '.join(loaded) if loaded else 'none'}",
        f"Not loaded: {', '.join(unloaded) if unloaded else 'none'}",
    ]
    return {"content": "\n".join(parts)}


def register(app):
    """Register /plugins command."""
    global _prev_send, _patched_send
    if hasattr(app, "register_plugin_command"):
        app.register_plugin_command(
            "/plugins",
            _handle_plugins,
            plugin_name="plugins_list_command",
            description="Show plugin load status",
        )
        return

    if _patched_send is not None:
        return
    _prev_send = app.send_message
    orig_send = _prev_send

    def patched(_self, text, *args, **kwargs):
        result = _handle_plugins(_self, text)
        if isinstance(result, dict) and result.get("handled") is False:
            if callable(orig_send):
                try:
                    return orig_send(text, *args, **kwargs)
                except Exception:
                    pass
            try:
                return type(_self).send_message(_self, text)
            except Exception:
                return {"error": "Original send_message handler is unavailable"}
        content = str(result.get("content", "") or "").strip()
        if content:
            _self._emit("message_added", {"role": "assistant", "content": content})
        return {"ok": True}

    _patched_send = types.MethodType(patched, app)
    app.send_message = _patched_send


def unregister(app):
    """Unregister /plugins command."""
    global _prev_send, _patched_send
    if hasattr(app, "unregister_plugin_command"):
        app.unregister_plugin_command("/plugins")
    if _patched_send is not None and app.send_message is _patched_send and _prev_send is not None:
        app.send_message = _prev_send
    _prev_send = None
    _patched_send = None
