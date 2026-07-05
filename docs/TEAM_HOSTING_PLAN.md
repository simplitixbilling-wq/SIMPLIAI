# SIMPLE_AI Team Hosting Plan
> For a 10-person team on a single Windows machine on a local network.

---

## Architecture Overview

```
[Team PCs] ──── LAN ──── [Host Machine :8765] ──── llama_cpp model
                               │
                         local_api_server.py
                         (binds 0.0.0.0 or subnet IP)
```

The host machine runs `agent_web.py` (or a headless variant). Everyone connects via their browser or Excel/VS Code using the host's LAN IP.

---

## Phase 1 — Network Access (Required)

**Goal:** Let teammates reach the API from their machines.

- [ ] Change `local_api_server.py` bind from `127.0.0.1` to `0.0.0.0` (or the host's static LAN IP, e.g. `192.168.1.100`)
- [ ] Open port `8765` in Windows Firewall (inbound TCP rule, restrict to LAN subnet)
- [ ] Assign a static IP to the host machine (via router DHCP reservation or Windows network settings)
- [ ] Optionally set a hostname alias: `simpleai.local` via router DNS or each PC's `hosts` file

---

## Phase 2 — Multi-User API Keys (Security)

**Goal:** Each user gets their own key so usage is traceable and roles can be managed.

- [ ] Extend `app_settings` (or a new `api_keys` table in `database.py`) to hold multiple named keys:
  ```json
  {
    "api_keys": {
      "alice": "key_abc123",
      "bob":   "key_def456"
    }
  }
  ```
- [ ] Update `local_api_server.py` auth check to validate against the key map
- [ ] Add an admin endpoint `POST /v1/admin/keys` (host-only, requires master key) to add/revoke keys
- [ ] Keys distributed to team members manually (Slack/email); each user sets it in Excel VBA config or VS Code env var `SIMPLE_AI_API_KEY`

---

## Phase 3 — Request Queue / Concurrency (Performance)

**Goal:** Prevent 10 people hammering the model simultaneously.

- [ ] Add a `threading.Semaphore(1)` (or `2` if host has enough RAM) in `local_api_server.py` around the `bridge.agent_chat()` call
- [ ] Return HTTP `429 Too Many Requests` with `Retry-After: 5` header when semaphore is full
- [ ] VBA / app.js client shows "AI is busy, please retry in a few seconds" on 429

---

## Phase 4 — Web UI for Team (Optional but Recommended)

**Goal:** Teammates open a browser and use the full chat UI without installing anything.

- [ ] Create `agent_server.py` — a headless server entry (no pywebview) that:
  - Serves `web/` folder as static files on port `8080`
  - Keeps `local_api_server.py` API on port `8765`
- [ ] Simple login page: one shared password or per-user HTTP Basic Auth via Nginx
- [ ] Each team member bookmarks `http://192.168.1.100:8080`

**Stack options (no external dependency):**
- Python `http.server` + serve `web/` statically (simplest)
- OR serve everything through port 8765 (add static file handler to `local_api_server.py`)

---

## Phase 5 — Reverse Proxy (Recommended for clean URLs + TLS)

**Goal:** Single clean URL, optional HTTPS on LAN.

- [ ] Install **Nginx for Windows** or **Caddy** on host
- [ ] Sample Nginx config:
  ```nginx
  server {
      listen 80;
      server_name simpleai.local;

      location / {
          root C:/Users/Chandana/Desktop_agent/web;
          index index.html;
      }

      location /v1/ {
          proxy_pass http://127.0.0.1:8765;
          proxy_set_header X-API-Key $http_x_api_key;
      }
  }
  ```
- [ ] For HTTPS: use `mkcert` to create a LAN cert, reference in Nginx ssl block

---

## Phase 6 — Auto-Start on Boot (Reliability)

**Goal:** App survives reboots without manual intervention.

Option A — Windows Task Scheduler (simplest):
- [ ] Create a task: trigger = At startup, action = `python agent_web.py`, run as SYSTEM or named account

Option B — NSSM (Non-Sucking Service Manager):
```
nssm install SimpleAI "C:\...\venv311_3.11\Scripts\python.exe" "C:\...\Desktop_agent\agent_web.py"
nssm set SimpleAI AppDirectory C:\...\Desktop_agent
nssm start SimpleAI
```

---

## Phase 7 — Excel & VS Code for Team

**Excel VBA:**
- Each user copies the `.xlsm` template or imports `integrations/excel/excel_vba_integration_full.txt`
- On first run: folder picker is prompted — they select **a shared network path** or their local copy
- The key file at that path is read automatically each time
- If host IP changes: users run `ResetSimpleAIAppPath()` macro and re-select

**VS Code:**
- Distribute `.vscode/tasks.json` and `vscode_fix_code.py` to each dev
- Each dev sets env var: `SIMPLE_AI_API_KEY=their_key`
- Update `--api-url` in tasks.json to point to host: `http://192.168.1.100:8765/v1/chat`

---

## Phase 8 — Backup & Logging

- [ ] Back up `chats.db` and `rag_databases/` to a network share or cloud nightly
- [ ] Add request logging in `local_api_server.py` (user key, timestamp, endpoint, response time) to a `api_access.log` file
- [ ] Rotate log weekly (Python `logging.handlers.RotatingFileHandler`)

---

## Summary Checklist

| Step | Priority | Effort |
|------|----------|--------|
| Bind to LAN IP + firewall rule | Must have | 15 min |
| Per-user API keys | Must have | 1–2 hrs |
| Request queue (semaphore) | Must have | 30 min |
| Auto-start on boot (NSSM) | Must have | 30 min |
| Headless web server (no pywebview) | Nice to have | 2–3 hrs |
| Nginx reverse proxy | Nice to have | 1 hr |
| Admin key management endpoint | Nice to have | 1–2 hrs |
| Backup + access logging | Nice to have | 1 hr |

---

## Implementation Order (Recommended)

1. **Phase 1** — get LAN access working and test from one other PC
2. **Phase 3** — add semaphore before anyone else connects
3. **Phase 2** — issue per-user keys
4. **Phase 6** — NSSM auto-start
5. **Phase 4/5** — web UI + proxy when team is onboarded
6. **Phase 8** — logging once live
