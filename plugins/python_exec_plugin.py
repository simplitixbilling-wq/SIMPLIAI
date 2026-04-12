"""Plugin: execute small Python snippets via /py command."""

PLUGIN_INFO = "Run safe Python snippets with /py"


def _format_result(result):
    if result.get("error"):
        parts = ["Python execution failed."]
        if result.get("stderr"):
            parts.append("\nStderr:\n" + str(result["stderr"]))
        if result.get("stdout"):
            parts.append("\nStdout:\n" + str(result["stdout"]))
        return "\n".join(parts)

    out = str(result.get("stdout", "")).strip()
    err = str(result.get("stderr", "")).strip()
    if not out and not err:
        return "Python executed successfully (no output)."

    msg = []
    if out:
        msg.append("Stdout:\n" + out)
    if err:
        msg.append("Stderr:\n" + err)
    return "\n\n".join(msg)


def handle_py(app, text: str):
    """Usage: /py <python_code>"""
    code = (text or "").split(None, 1)
    if len(code) < 2 or not code[1].strip():
        return {
            "content": (
                "Usage: /py <python code>\n"
                "Example: /py print(sum([1,2,3]))"
            )
        }

    result = app.execute_python_snippet(code[1], timeout=8)
    return {"content": _format_result(result)}


def register(app):
    app.register_plugin_command(
        "/py",
        handle_py,
        plugin_name="python_exec_plugin",
        description="Run restricted Python snippets",
    )


def unregister(app):
    app.unregister_plugin_command("/py")
