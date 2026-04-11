"""Sample plugin: adds /hello command."""

import types

PLUGIN_INFO = "Sample: /hello command"

_prev_send = None
_patched_send = None


def _handle_hello(app, text: str):
    if str(text or "").strip().lower() != "/hello":
        return {"handled": False}
    return {"content": "Hello from plugin. Try /roll 2d6 or /plugins."}


def register(app):
    """Register /hello command."""
    global _prev_send, _patched_send
    if hasattr(app, "register_plugin_command"):
        app.register_plugin_command(
            "/hello",
            _handle_hello,
            plugin_name="hello_command_plugin",
            description="Simple hello command",
        )
        return

    # Backward-compatible fallback for older Bridge implementations.
    if _patched_send is not None:
        return
    _prev_send = app.send_message
    orig_send = _prev_send

    def patched(_self, text, *args, **kwargs):
        result = _handle_hello(_self, text)
        if isinstance(result, dict) and result.get("handled") is False:
            if callable(orig_send):
                try:
                    return orig_send(text, *args, **kwargs)
                except Exception:
                    pass
            # Hard fallback: invoke Bridge.send_message directly from class.
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
    """Unregister /hello command."""
    global _prev_send, _patched_send
    if hasattr(app, "unregister_plugin_command"):
        app.unregister_plugin_command("/hello")
    if _patched_send is not None and app.send_message is _patched_send and _prev_send is not None:
        app.send_message = _prev_send
    _prev_send = None
    _patched_send = None
