"""Sample plugin: adds /roll NdM dice command."""

import random
import re
import types

PLUGIN_INFO = "Sample: /roll NdM dice command"

_prev_send = None
_patched_send = None

_ROLL_RE = re.compile(r"^/roll\s*(?:(\d+)d)?(\d+)$", re.IGNORECASE)


def _handle_roll(app, text: str):
    msg = str(text or "").strip()
    match = _ROLL_RE.match(msg)
    if not match:
        return {"handled": False}

    count = int(match.group(1) or 1)
    sides = int(match.group(2))

    if count < 1 or count > 50 or sides < 2 or sides > 1000:
        return {"content": "Use /roll NdM with limits: N=1..50, M=2..1000"}

    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls)
    return {
        "content": f"Rolled {count}d{sides}: {rolls}  Total={total}",
        "rolls": rolls,
        "total": total,
    }


def register(app):
    """Register /roll command."""
    global _prev_send, _patched_send
    if hasattr(app, "register_plugin_command"):
        app.register_plugin_command(
            "/roll",
            _handle_roll,
            plugin_name="dice_roll_plugin",
            description="Roll dice with /roll NdM",
        )
        return

    if _patched_send is not None:
        return
    _prev_send = app.send_message
    orig_send = _prev_send

    def patched(_self, text, *args, **kwargs):
        result = _handle_roll(_self, text)
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
    """Unregister /roll command."""
    global _prev_send, _patched_send
    if hasattr(app, "unregister_plugin_command"):
        app.unregister_plugin_command("/roll")
    if _patched_send is not None and app.send_message is _patched_send and _prev_send is not None:
        app.send_message = _prev_send
    _prev_send = None
    _patched_send = None
