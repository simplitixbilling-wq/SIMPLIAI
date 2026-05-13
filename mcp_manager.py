"""
mcp_manager.py  –  MCP (Model Context Protocol) server manager for SIMPLE_AI.

Manages connections to MCP-compatible tool servers (stdio & SSE transport)
and exposes their tools to the AI chat via the existing tool-call system.
"""

import json
import logging
import os
import subprocess
import threading
import time
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Single MCP Server Connection
# ═══════════════════════════════════════════════════════════════

class MCPServerConnection:
    """Manages a JSON-RPC 2.0 connection to one MCP server."""

    def __init__(self, config: dict):
        self.config = config
        self.name: str = config["name"]
        self.transport: str = config.get("transport", "stdio")
        self.process: Optional[subprocess.Popen] = None
        self.connected: bool = False
        self.tools: list = []
        self.server_info: dict = {}
        self._msg_id: int = 0
        self._lock = threading.Lock()

    # ── JSON-RPC helpers ──────────────────────────────────

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _write_msg(self, msg: dict):
        data = json.dumps(msg, separators=(",", ":")) + "\n"
        self.process.stdin.write(data.encode("utf-8"))
        self.process.stdin.flush()

    def _read_msg(self, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                if self.process.poll() is not None:
                    stderr_out = ""
                    try:
                        stderr_out = self.process.stderr.read().decode("utf-8", errors="replace")[:500]
                    except Exception:
                        pass
                    raise ConnectionError(
                        f"MCP server '{self.name}' exited unexpectedly. stderr: {stderr_out}"
                    )
                continue
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                log.debug("MCP %s: skip non-JSON: %s", self.name, line[:200])
                continue
        raise TimeoutError(f"MCP server '{self.name}' did not respond within {timeout}s")

    def _request(self, method: str, params: dict = None) -> dict:
        msg_id = self._next_id()
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        with self._lock:
            self._write_msg(msg)
            for _ in range(50):
                resp = self._read_msg()
                if resp.get("id") == msg_id:
                    if "error" in resp:
                        err = resp["error"]
                        raise RuntimeError(
                            f"MCP error ({err.get('code','?')}): {err.get('message','Unknown')}"
                        )
                    return resp.get("result", {})
        raise RuntimeError(f"No matching response from '{self.name}' for id={msg_id}")

    def _notify(self, method: str, params: dict = None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        with self._lock:
            self._write_msg(msg)

    # ── Connection lifecycle ──────────────────────────────

    def connect(self) -> dict:
        if self.transport == "stdio":
            return self._connect_stdio()
        elif self.transport == "sse":
            return self._connect_sse()
        raise ValueError(f"Unsupported transport: {self.transport}")

    def _connect_stdio(self) -> dict:
        cmd = self.config.get("command", "")
        args = self.config.get("args", [])
        if not cmd:
            raise ValueError("No command specified for stdio server")

        full_cmd = [cmd] + args
        env_extra = self.config.get("env", {})
        env = {**os.environ, **env_extra}

        log.info("MCP: starting %s → %s", self.name, full_cmd)
        kwargs = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self.process = subprocess.Popen(full_cmd, **kwargs)

        # Initialize handshake
        result = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "SIMPLE_AI", "version": "1.0.0"},
        })
        self.server_info = result.get("serverInfo", {})
        self._notify("notifications/initialized")

        # List available tools
        tools_result = self._request("tools/list", {})
        self.tools = tools_result.get("tools", [])
        self.connected = True
        log.info("MCP: %s connected — %d tools", self.name, len(self.tools))
        return {"server_info": self.server_info, "tools": self.tools}

    def _connect_sse(self) -> dict:
        """SSE transport — POST JSON-RPC to the server URL."""
        import httpx

        url = self.config.get("url", "")
        if not url:
            raise ValueError("No URL specified for SSE server")
        headers = self.config.get("headers", {})

        with httpx.Client(timeout=30) as client:
            # Initialize
            resp = client.post(url, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "SIMPLE_AI", "version": "1.0.0"},
                }
            }, headers=headers)
            resp.raise_for_status()
            init_data = resp.json()
            self.server_info = init_data.get("result", {}).get("serverInfo", {})

            # List tools
            resp = client.post(url, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
            }, headers=headers)
            resp.raise_for_status()
            tools_data = resp.json()
            self.tools = tools_data.get("result", {}).get("tools", [])

        self.connected = True
        self._sse_url = url
        self._sse_headers = headers
        log.info("MCP: %s (SSE) connected — %d tools", self.name, len(self.tools))
        return {"server_info": self.server_info, "tools": self.tools}

    def disconnect(self):
        self.connected = False
        self.tools = []
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        log.info("MCP: %s disconnected", self.name)

    # ── Tool execution ────────────────────────────────────

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self.connected:
            raise ConnectionError(f"Server '{self.name}' is not connected")

        if self.transport == "stdio":
            result = self._request("tools/call", {
                "name": tool_name, "arguments": arguments,
            })
        elif self.transport == "sse":
            import httpx
            payload = {
                "jsonrpc": "2.0", "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            with httpx.Client(timeout=60) as client:
                resp = client.post(self._sse_url, json=payload, headers=self._sse_headers)
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(data["error"].get("message", "Unknown error"))
                result = data.get("result", {})
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")

        content = result.get("content", [])
        texts = []
        for item in content:
            t = item.get("type", "")
            if t == "text":
                texts.append(item.get("text", ""))
            elif t == "image":
                texts.append(f"[Image: {item.get('mimeType', 'image')}]")
            else:
                texts.append(json.dumps(item))
        return "\n".join(texts) if texts else json.dumps(result)


# ═══════════════════════════════════════════════════════════════
#  MCP Manager — orchestrates multiple servers
# ═══════════════════════════════════════════════════════════════

class MCPManager:
    """Central manager for all MCP server connections."""

    def __init__(self, db):
        self.db = db
        self._configs: Dict[str, dict] = self._load_configs()
        self._servers: Dict[str, MCPServerConnection] = {}

    # ── Persistence ───────────────────────────────────────

    def _load_configs(self) -> dict:
        raw = self.db.get_kv("mcp_servers")
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_configs(self):
        self.db.set_kv("mcp_servers", json.dumps(self._configs))

    # ── Config CRUD ───────────────────────────────────────

    def get_server_list(self) -> list:
        servers = []
        for name, cfg in self._configs.items():
            srv = self._servers.get(name)
            servers.append({
                "name": name,
                "transport": cfg.get("transport", "stdio"),
                "command": cfg.get("command", ""),
                "args": cfg.get("args", []),
                "url": cfg.get("url", ""),
                "enabled": cfg.get("enabled", True),
                "connected": bool(srv and srv.connected),
                "tool_count": len(srv.tools) if srv and srv.connected else 0,
            })
        return servers

    def add_server(self, config: dict) -> dict:
        name = config.get("name", "").strip()
        if not name:
            return {"error": "Server name is required"}
        if name in self._configs:
            return {"error": f"Server '{name}' already exists"}
        self._configs[name] = config
        self._save_configs()
        return {"ok": True}

    def update_server(self, old_name: str, config: dict) -> dict:
        if old_name not in self._configs:
            return {"error": f"Server '{old_name}' not found"}
        if old_name in self._servers:
            self._servers[old_name].disconnect()
            del self._servers[old_name]
        new_name = config.get("name", old_name)
        if new_name != old_name:
            del self._configs[old_name]
        self._configs[new_name] = config
        self._save_configs()
        return {"ok": True}

    def remove_server(self, name: str) -> dict:
        if name in self._servers:
            self._servers[name].disconnect()
            del self._servers[name]
        if name in self._configs:
            del self._configs[name]
            self._save_configs()
            return {"ok": True}
        return {"error": f"Server '{name}' not found"}

    # ── Connection management ─────────────────────────────

    def connect_server(self, name: str) -> dict:
        cfg = self._configs.get(name)
        if not cfg:
            return {"error": f"Server '{name}' not configured"}
        if name in self._servers:
            self._servers[name].disconnect()
        try:
            srv = MCPServerConnection(cfg)
            result = srv.connect()
            self._servers[name] = srv
            return {"ok": True, **result}
        except Exception as e:
            log.exception("MCP: connect failed for %s", name)
            return {"error": str(e)}

    def disconnect_server(self, name: str) -> dict:
        if name in self._servers:
            self._servers[name].disconnect()
            del self._servers[name]
            return {"ok": True}
        return {"error": f"Server '{name}' not connected"}

    # ── Tool access ───────────────────────────────────────

    def get_all_tools(self) -> list:
        tools = []
        for name, srv in self._servers.items():
            if srv.connected:
                for tool in srv.tools:
                    tools.append({
                        "server": name,
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "inputSchema": tool.get("inputSchema", {}),
                    })
        return tools

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        srv = self._servers.get(server_name)
        if not srv or not srv.connected:
            return {"error": f"Server '{server_name}' not connected"}
        try:
            result = srv.call_tool(tool_name, arguments)
            return {"ok": True, "result": result}
        except Exception as e:
            log.exception("MCP: tool call failed %s/%s", server_name, tool_name)
            return {"error": str(e)}

    def build_tool_definitions(self) -> str:
        """Build prompt text describing available MCP tools for the AI."""
        tools = self.get_all_tools()
        if not tools:
            return ""

        lines = ["\n\n--- External MCP Tools ---"]
        for t in tools:
            schema = t.get("inputSchema", {})
            props = schema.get("properties", {})
            required = set(schema.get("required", []))

            param_parts = []
            for pname, pinfo in props.items():
                req = "(required)" if pname in required else "(optional)"
                desc = pinfo.get("description", pinfo.get("type", ""))
                param_parts.append(f"    {pname} {req}: {desc}")

            params_str = "\n".join(param_parts) if param_parts else "    (no parameters)"
            lines.append(
                f"\n  Tool: mcp:{t['server']}:{t['name']}\n"
                f"  Description: {t['description']}\n"
                f"  Parameters:\n{params_str}"
            )

        lines.append(
            "\nTo use an MCP tool, output:\n"
            '<tool_call>{"tool": "mcp:<server>:<tool_name>", "args": {"param": "value"}}</tool_call>'
        )
        return "\n".join(lines)

    # ── Cleanup ───────────────────────────────────────────

    def shutdown(self):
        for name in list(self._servers.keys()):
            try:
                self._servers[name].disconnect()
            except Exception:
                pass
        self._servers.clear()
