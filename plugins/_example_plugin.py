"""
Example SIMPLE_AI Plugin
========================
Place .py files in the plugins/ folder. Each must define register(app).

Available hooks:
  - app.root          : the CTk root window
  - app.chats         : dict of all chats
  - app.model         : the loaded Llama model (or None)
  - app._show_toast() : show a notification
  - app.plugin_manager: access to other plugins

Optional:
  PLUGIN_INFO = "Short description shown in UI"
  def unregister(app): called when plugin is unloaded
"""

PLUGIN_INFO = "Example plugin — rename without _ prefix to activate"


def register(app):
    """Called once when the plugin is loaded."""
    print("[PLUGIN] Example plugin loaded")


def unregister(app):
    """Called when the plugin is unloaded (optional)."""
    print("[PLUGIN] Example plugin unloaded")
