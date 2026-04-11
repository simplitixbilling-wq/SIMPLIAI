# SIMPLE_AI — Improvement Roadmap

## Performance
1. ✅ **GPU acceleration** — rebuild llama-cpp-python with CUDA support (`n_gpu_layers=-1`), 5-10x speed boost
2. ✅ **Streaming response buffer** — batch token renders (every 3-5 tokens) instead of per-token UI update, reduces flicker
3. ✅ **Async model loading** — load model in background thread so UI doesn't freeze on startup
4. ✅ **RAG cache** — cache TF-IDF vectors to disk so re-opening an existing database is instant

## Model Management
5. ✅ **HuggingFace model downloader** — search and download `.gguf` files from within the app (RAM/VRAM compatibility filter)
6. ✅ **Hot-swap models** — switch models without restarting (unload → gc.collect → cuda.empty_cache → load new)
7. ✅ **Model metadata display** — show quantization type and file size in the dropdown (e.g. `Q5_K_S · 3.2GB`)
8. ✅ **Per-model settings** — save temperature/context/threads per model (`model_configs.json`)

## Chat & UI
9. ✅ **Chat search** — search across all saved chats by keyword
10. ✅ **Export chat** — save conversation as `.txt` or `.pdf`
11. ✅ **Chat folders/tags** — organize chats into categories
12. ✅ **Markdown rendering** — render `**bold**`, code blocks, bullet lists in chat bubbles properly
13. ✅ **Syntax highlighting** — highlight code blocks in responses (` ```python ` → colored)
14. ✅ **Regenerate response** — button to re-run last prompt with different output
15. ✅ **Edit sent message** — click a user message to edit and resend
16. ✅ **Token counter** — show live token count in context window
17. ✅ **Stop generation button** — actually cancel mid-stream (set a threading Event flag)

## RAG
18. ✅ **RAG database per chat** — assign different databases to different chats (`_rag_settings.json`)
19. ✅ **Chunk preview** — show which source chunks were used in the answer (citations)
20. ✅ **Web page import** — paste a URL, scrape and add to RAG database (BeautifulSoup)
21. ✅ **Re-index button** — UI 🔄 button to re-process documents without deleting the database
22. ✅ **Hybrid search** — add BM25 alongside TF-IDF for better keyword recall (6-signal retrieval)
23. ✅ **Chunk size control** — slider in UI to set chunk size / overlap when adding documents

## System / Reliability
24. ✅ **Settings panel** — GUI for temperature, top_p, repeat_penalty, max response tokens (`app_settings.json`)
25. ✅ **Auto-save timer** — save all chats every 60 seconds automatically
26. ✅ **Error toasts** — non-blocking popup notifications that auto-dismiss (info/warn/error)
27. ✅ **First-run wizard** — detect no models found → guide user to download or open models folder
28. ✅ **Memory/RAM monitor** — warn before loading a model that exceeds available RAM (confirmation dialog)
29. ✅ **Custom system prompt per chat** — editable system prompt stored with each chat (`_system_prompts.json`)

## Distribution
30. **Auto-updater** — check GitHub releases for a new `.exe` version and prompt user
31. **App icon** — proper `.ico` file for the exe and taskbar
32. **Installer (NSIS/Inno Setup)** — proper Windows installer with Start Menu shortcut

---

## Progress: 40/48 complete ████████████████░░░░ 83%

| Section | Status |
|---|---|
| Performance (#1–4) | ✅ 4/4 |
| Model Management (#5–8) | ✅ 4/4 |
| Chat & UI (#9–17) | ✅ 9/9 |
| RAG (#18–23) | ✅ 6/6 |
| System / Reliability (#24–29) | ✅ 6/6 |
| Distribution (#30–32) | ⬜ 0/3 |
| API & Integration (#33–37) | ⬜ 0/5 |
| Architecture (#38–41) | ✅ 4/4 |
| Advanced UX (#42–48) | ✅ 7/7 |

---

## API & Integration — *the biggest gap vs. Ollama/LM Studio*
33. **Local API server** — Flask/FastAPI `/v1/chat/completions` (OpenAI-compatible) so VS Code, scripts, and other apps can call SIMPLE_AI like Ollama
34. **Streaming SSE endpoint** — server-sent events streaming for the API (matches OpenAI `stream: true` spec)
35. **Multi-format model import** — accept SafeTensors/HF repos and auto-convert to GGUF using `llama.cpp/convert` scripts
36. **OpenAI JSON export** — export chats in OpenAI messages format (`[{role, content}]`) for fine-tuning / sharing
37. **Webhook / callback support** — POST completed responses to a configurable URL (automation pipelines)

## Architecture — *single-file → modular*
38. ✅ **Split into modules** — `ui_components.py`, `model_manager.py`, `chat_manager.py`, `settings_manager.py`, `generation.py`, `rag_handler.py` + standalone `utils.py`, `database.py`, `plugin_manager.py`, `system_tools.py` (keep `agent.py` as thin entry point)
39. ✅ **Plugin system** — `plugins/` folder, each plugin is a Python file with `register(app)` hook — add tools, RAG sources, or UI panels without editing core code
40. ✅ **SQLite chat storage** — replace individual JSON files with a single `chats.db` — faster search, atomic writes, no corruption on crash
41. ✅ **Config migration** — auto-detect old JSON configs and migrate to new structure on first run after update

## Advanced UX — *match LM Studio / Jan.ai polish*
42. ✅ **Real-time markdown rendering** — render bold, italic, lists, headers, tables live during streaming (tag-based textbox formatting, not post-hoc regex)
43. ✅ **Conversation branching** — fork any message to explore alternate responses (tree view, like ChatGPT "edit & regenerate" but keeping both)
44. ✅ **Multi-model compare** — send same prompt to 2 models side-by-side, compare outputs (A/B testing)
45. ✅ **Image input (multimodal)** — support vision models (LLaVA, etc.) — paste/upload image, model describes or answers about it
46. ✅ **Voice input/output** — speech-to-text input + TTS for responses (whisper.cpp + local TTS)
47. ✅ **Context window visualizer** — live bar showing how much of the model's context is used (prompt + history + RAG + response budget)
48. ✅ **Keyboard shortcuts** — `Ctrl+N` new chat, `Ctrl+L` load model, `Ctrl+Shift+S` settings, `Ctrl+E` export, `Esc` stop generation

---

## What gets us to 8.5+/10 (minimum viable competitive set)

| # | Feature | Why it matters | Effort |
|---|---------|----------------|--------|
| 33 | Local API server | Ecosystem compatibility — this alone closes 50% of the gap with Ollama | Medium |
| 38 | ✅ Split into modules | Maintainability — reviewers/contributors can actually work on it | Medium |
| 42 | Real-time markdown | Visual polish — the #1 thing users notice in LM Studio | High |
| 47 | Context window visualizer | Unique feature — even LM Studio's is basic | Low |
| 48 | Keyboard shortcuts | Table-stakes UX every desktop app needs | Low |
| 45 | Image input (multimodal) | Future-proofing — vision models are becoming standard | High |

---

## Priority Picks (Highest ROI)
| # | Feature | Impact | Status |
|---|---------|--------|--------|
| 1 | GPU acceleration | Speed: CPU 10 t/s → GPU 60+ t/s | ✅ |
| 5 | HuggingFace model downloader | No more manual file management | ✅ |
| 13 | Syntax highlighting | Code responses become readable | ✅ |
| 14 | Regenerate response | Core UX expected by users | ✅ |
| 24 | Settings panel | Exposes power-user controls | ✅ |
| 33 | Local API server | Ecosystem compat — the #1 missing feature | ⬜ |
| 42 | Real-time markdown | Visual polish gap vs. LM Studio | ⬜ |
| 47 | Context window visualizer | Unique differentiator, low effort | ⬜ |
