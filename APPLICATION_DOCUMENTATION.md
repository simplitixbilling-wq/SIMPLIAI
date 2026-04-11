# SIMPLE_AI - Complete Application Documentation

## 1. Application Overview
SIMPLE_AI is a Windows desktop AI assistant application built with:
- Python backend
- pywebview desktop window
- HTML/CSS/JavaScript frontend
- SQLite persistence
- Local GGUF model inference via `llama-cpp-python`

It supports two primary modes:
1. Chat mode for conversational use (with optional RAG, web search, image/file input)
2. Agent mode for instruction-driven file processing and output generation


## 2. Core Architecture

### 2.1 Entry Point
- Main launcher: `agent_web.py`
- Creates pywebview window and binds backend API (`Bridge`) to frontend JavaScript.

### 2.2 Backend Core
- Main API class: `Bridge` in `bridge.py`
- Handles model lifecycle, chat workflows, RAG workflows, exports, settings, file processing, monitoring, and frontend events.

### 2.3 Frontend
- Layout and modals: `web/index.html`
- Logic/events/API calls: `web/app.js`
- Styling/themes/layout: `web/style.css`

### 2.4 Data Persistence
- SQLite DB: `chats.db`
- JSON/template files: `instruction_templates.json` and migrated/legacy JSON sources
- Generated output folders: `exports/`, `processed_files/`, `rag_databases/`


## 3. Folder-by-Folder Purpose

- `agent_web.py`: Desktop app launcher
- `bridge.py`: Main backend API exposed to UI
- `database.py`: SQLite helpers and data access
- `chat_manager.py`: Chat utility logic
- `model_manager.py`: Model selection/load helpers
- `generation.py`: Generation-related helpers
- `rag_manager.py`: RAG indexing + retrieval
- `rag_handler.py`: RAG orchestration utilities
- `plugin_manager.py`: Plugin discovery and reload
- `settings_manager.py`: Settings helper logic
- `system_tools.py`: System-level utilities
- `web/`: Frontend files
- `models/`: GGUF model files
- `rag_databases/`: Knowledge database folders
- `saved_chats/`: Legacy chat JSONs / backups
- `exports/`: Exported chat/output files
- `processed_files/`: Agent-generated analysis output files
- `webview_data/`: Browser cache/storage for webview runtime


## 4. Startup and Initialization Flow

1. `agent_web.py` starts
2. `Bridge()` is created
3. Bridge initialization loads:
   - GPU/RAM detection
   - runtime config
   - saved settings from DB
   - chats + metadata
   - model configs
   - system prompts
   - RAG manager
   - plugins
4. pywebview window opens `web/index.html`
5. Frontend waits for API readiness
6. Frontend runs initialization:
   - app info (theme/status)
   - models list
   - model status
   - chats list
   - starts monitor updates


## 5. Complete UI Guide and How to Use Every Part

## 5.1 Top Bar

### Sidebar Toggle (☰)
- Show/hide left sidebar.

### Mode Buttons
- `Chat`: Opens regular conversation area.
- `Agent`: Opens split-screen instruction/file-processing area.

### Model Dropdown
- Lists `.gguf` models found in `models/`.
- Select one model before loading.

### Load/Unload Button
- `Load` initializes selected model.
- After loading, button becomes `Unload`.

### Model Settings (⚙)
- Opens per-model configuration modal.
- Set model-specific context/temperature/threads.

### System Prompt (📝)
- Opens system prompt editor for current chat.
- Stored per chat.

### HuggingFace Download (⬇)
- Opens downloader modal.
- Search GGUF models and download into `models/`.

### Reload (🔄, dev use)
- Reload frontend (visible in development usage paths).

### Header Status Pill
- Shows runtime mode + memory summary.
- Active pulse indicates live/processing states.


## 5.2 Sidebar

### Model Status Area
- Shows loaded model name.
- Progress bar during model load.
- Context usage indicator.

### + New Chat
- Creates a new chat thread.

### Chat Search
- Filters chat list instantly.

### Chat List Item Actions
- Select chat
- Rename chat
- Delete chat
- Export chat

### Knowledge Section (RAG)
- Displays available RAG databases.
- Select one database per chat.
- Create/delete/reindex databases.

### Settings Button
- Opens global settings modal.

### CPU/RAM Monitor Footer
- Live monitor from backend thread.


## 5.3 Chat Mode (Main Conversation)

### Message List
- User messages and assistant messages with markdown rendering.

### Input Controls
- Text area with auto-resize
- Enter sends (Shift+Enter for newline)
- Token estimate counter

### Action Icons Near Input
- Web search toggle
- Upload file/document
- Attach image
- Voice input/action (if configured)

### Stop Button
- Interrupts active generation safely.

### Assistant Message Actions
- Copy
- Regenerate
- Branch new chat from point
- Speak text
- Export specific output


## 5.4 Agent Mode (Instruction + File Processing)

The agent area is split into left and right panels.

### Left Panel - Instructions
- Create instruction templates
- Edit/delete templates
- Click template to open Agent Chat state

### Left Panel - Output Files
- Lists files in `processed_files/`
- Open file directly
- Delete file directly
- Refresh list button

### Right Panel States

#### State A: Empty
- Prompt to select or create instruction.

#### State B: Create/Edit Instruction
Fields:
- Name
- Instruction text
- Default output format (excel/csv/pdf/txt)

Save behavior:
- Stores template
- Opens agent chat for that template

#### State C: Agent Chat
- Shows active instruction header
- Optional file attachments
- Optional additional notes
- Sends request to backend processing APIs

If files are attached:
- Uses `process_files_with_ai(...)`
- Generates output file in selected format
- Shows assistant response + output actions

If only text:
- Uses `agent_chat(...)`
- Returns direct assistant response


## 5.5 Settings Modal - How to Use

Fields include:
- Theme (Dark/Light)
- Temperature
- Top-p
- Repeat penalty
- Max response tokens
- Hardware/system info display

Save behavior:
- Applies theme immediately
- Persists theme and settings to DB
- Theme is restored automatically on next app open


## 5.6 Per-Model Settings Modal

Set values per model filename:
- Temperature
- CPU threads
- Context window (`n_ctx`)

Use this when one model needs different performance/quality settings than others.


## 5.7 System Prompt Modal

- Set behavior instructions for current chat only.
- Example use: force concise style or domain constraints.


## 5.8 HuggingFace Model Download Modal

Workflow:
1. Search model keyword
2. Review GGUF variants, size, and fit status
3. Download selected file
4. File is saved under `models/`
5. Refresh models list and load


## 5.9 RAG Create Modal

You can create knowledge DB from folder/URL.

Fields:
- Database name
- Source path/URL
- Chunk size
- Overlap

Result:
- Builds indexed knowledge under `rag_databases/<name>/`
- Select DB in sidebar for retrieval-enhanced responses


## 5.10 Compare Models Modal

- Pick model A and model B
- Enter one prompt
- Receive side-by-side outputs for evaluation


## 5.11 Plugins Modal

- Lists detected plugins from `plugins/`
- Reload plugins without restarting app


## 5.12 Export Modal

Export chat or output in:
- TXT
- PDF
- DOCX
- CSV
- XLSX

Saved into `exports/` (or chosen path where applicable).


## 5.13 Detailed Step-by-Step Usage for Every UI Input Option

This section is a practical operator guide. It explains what each input does, how to use it correctly, and what output to expect.

### 5.13.1 Top Bar Inputs

### A) Model Dropdown
Purpose:
- Select which local GGUF model will be used for chat and agent processing.

Steps:
1. Put one or more `.gguf` files in `models/`.
2. Open app.
3. Click model dropdown.
4. Select desired model.
5. Click `Load`.

Expected result:
- Model label appears in sidebar.
- Context bar appears.
- Status shows loaded context and runtime mode.

When to change it:
- Use smaller quant models for low RAM.
- Use bigger models for better quality if RAM/VRAM allows.


### B) Load / Unload Button
Purpose:
- Load selected model into memory or unload current model to free RAM/VRAM.

Steps:
1. Select model from dropdown.
2. Click `Load`.
3. Wait for load progress and completion.
4. To free memory, click `Unload`.

Expected result:
- `Load` changes to `Unload`.
- CPU/RAM usage may increase while loaded.


### C) Chat / Agent Mode Buttons
Purpose:
- Switch between conversational mode and instruction/file-processing mode.

Steps:
1. Click `Chat` for normal conversation workflows.
2. Click `Agent` for file analysis workflows.

Expected result:
- Corresponding panel becomes visible.


### D) Model Settings Button
Purpose:
- Save per-model generation settings.

Inputs:
- Temperature
- Threads
- Context size (`n_ctx`)

Steps:
1. Ensure a model is selected.
2. Click model settings button.
3. Set values.
4. Save.

Expected result:
- Values are stored per model filename and reused later.


### E) System Prompt Button
Purpose:
- Define assistant behavior for current chat.

Steps:
1. Open a chat.
2. Click system prompt button.
3. Enter instruction such as style, structure, role.
4. Save.

Expected result:
- Prompt is applied on next responses for that chat.


### F) HuggingFace Download Button
Purpose:
- Search and download GGUF models from HuggingFace.

Inputs:
- Search query
- Model variant/file selection

Steps:
1. Click HuggingFace button.
2. Type model family name (example: `qwen`, `gemma`, `llama`).
3. Review file size and fit indicators.
4. Choose a compatible quantized GGUF file.
5. Download.

Expected result:
- File appears in `models/`.
- It appears in model dropdown after refresh/reopen.


## 5.13.2 Sidebar Inputs

### A) + New Chat
Purpose:
- Start a fresh conversation context.

Steps:
1. Click `+ New Chat`.
2. New chat is created and becomes selectable.

Expected result:
- Chat list includes new entry (example: `Chat 12`).


### B) Chat Search Input
Purpose:
- Quickly find chats by name.

Steps:
1. Type part of chat name.
2. Matching chats remain visible.


### C) Chat Row Actions (Rename/Delete/Export)
Rename steps:
1. Click rename icon.
2. Enter new name.
3. Press Enter.

Delete steps:
1. Click delete icon.
2. Confirm.

Export steps:
1. Click export icon.
2. Choose format.
3. Save.


### D) Knowledge (RAG) Selector
Purpose:
- Attach a knowledge base to current chat.

Steps:
1. Create or select RAG database.
2. Click database name in sidebar.
3. Ask domain questions in chat.

Expected result:
- Responses include context-aware information based on selected RAG content.


## 5.13.3 Chat Input Area - Every Control

### A) Main Message Textarea
Purpose:
- Primary input for user messages.

Steps:
1. Type question/request.
2. Press Enter to send.
3. Use Shift+Enter for newline.


### B) Web Toggle Icon
Purpose:
- Enable web search-assisted response mode.

When ON:
- App may fetch web snippets before answering.

When OFF:
- Response is based on model + local context only.


### C) Upload Document Icon
Purpose:
- Attach document content for one query.

Supported sources (through extractor pipeline):
- TXT, CSV, PDF, XLSX, DOCX

Steps:
1. Click upload icon.
2. Select file.
3. Send your query describing what to do with file.

Expected result:
- Extracted content is included in prompt context.


### D) Attach Image Icon
Purpose:
- Provide image input for multimodal/vision flow where supported.

Steps:
1. Click image icon.
2. Select image.
3. Ask image-specific question.

Expected result:
- If multimodal resources are available, model uses them.
- Otherwise, fallback handling is used.


### E) Stop Button
Purpose:
- Interrupt generation safely.

Steps:
1. Send prompt.
2. While generating, click `Stop`.

Expected result:
- Generation halts.
- Partial output remains visible.


### F) Assistant Response Action Buttons
Copy:
- Copies response text.

Regenerate:
- Re-runs response for same query.

Branch:
- Creates new chat from selected point.

Speak:
- Uses TTS to read response.

Export:
- Exports selected response or chat content.


## 5.13.4 Agent Mode - Full Workflow (Create Instruction -> Upload File/Chat -> Get Output)

### Step 1: Open Agent Mode
1. Click `Agent` in top bar.
2. Left panel shows `Instructions` and `Output Files`.

### Step 2: Create Instruction Template
1. Click `+ New` under Instructions.
2. Fill fields:
    - Name: short unique label (example: `Billing Reconciliation`)
    - Instructions: exact processing logic and expected checks
    - Default Output Format: excel/csv/pdf/txt
3. Click `Save Instruction`.

Best practice for instruction text:
- Mention exact columns to compare.
- Ask for final results only, not code.
- Ask for discrepancy table format.

Example instruction:
- "Compare billing and sales files by order id. Find missing IDs, quantity mismatches, price mismatches, and total amount mismatches. Return summary counts and a detailed discrepancy table. Do not return code."

### Step 3: Open Instruction Chat
1. Click saved instruction card.
2. Right panel switches to agent chat state.

### Step 4A: File Processing Flow
1. Click paperclip icon and upload one or more files.
2. Optionally type additional note in agent input.
3. Click send.

Backend behavior:
- Combines saved instruction + additional note + file contents.
- Runs AI processing.
- Generates output file in selected format.

Expected result in chat:
- AI response text (analysis result)
- Output file actions:
   - Open file
   - Delete file

Output location:
- `processed_files/analysis_<timestamp>.<ext>`

### Step 4B: Text-only Agent Chat Flow
1. Do not upload files.
2. Type query in agent input.
3. Send.

Expected result:
- Uses `agent_chat` and returns direct answer text.

### Step 5: Manage Generated Outputs
From Output Files section (left bottom):
1. Open icon: launch file with system default app.
2. Delete icon: remove file permanently.
3. Refresh icon: reload latest files from disk.


## 5.13.5 Settings Modal - Input-by-Input Guide

### Theme
Values:
- Dark
- Light

Use:
1. Select theme.
2. Save settings.

Result:
- UI theme changes and persists across restart.


### Temperature
Purpose:
- Controls creativity/randomness.

Lower values:
- More deterministic and repeatable.

Higher values:
- More diverse output.


### Top-p
Purpose:
- Nucleus sampling cutoff.

Lower top-p:
- Safer, narrower token choice.

Higher top-p:
- Broader variation.


### Repeat Penalty
Purpose:
- Reduces repetitive looping text.


### Max Response Tokens
Purpose:
- Caps output length.

Tip:
- Keep moderate value for speed and concise responses.


## 5.13.6 RAG Create Modal - Input-by-Input Guide

### Name
- Unique identifier of knowledge DB.

### Source Path / URL
- Folder path for local files or URL for web source indexing.

### Chunk Size
- Amount of text per chunk during indexing.

Guideline:
- 300-700 is good for most business documents.

### Overlap
- Shared text between adjacent chunks.

Guideline:
- 50-150 prevents context cutoffs at boundaries.


## 5.13.7 Compare Models Modal - Input-by-Input Guide

Inputs:
- Model A
- Model B
- Prompt

Steps:
1. Select two models.
2. Enter identical prompt.
3. Run compare.

Use cases:
- Quality benchmarking
- Speed vs quality tradeoff checks
- Prompt robustness checks


## 5.13.8 Plugins - What It Is, How It Works, and Sample

What plugins are:
- Optional Python modules loaded from `plugins/`.
- Used to extend app behavior without editing core files.

How plugin loading works:
1. App startup scans `plugins/` for Python files.
2. Each plugin should expose a `register(app)` function.
3. Plugin manager imports plugin and calls `register(app)`.
4. Plugin appears in plugin list modal.
5. You can reload plugins from UI without full restart.

Typical plugin use cases:
- Add custom command handlers
- Add utility functions
- Add domain-specific helpers
- Hook additional logging/workflows

Minimal sample plugin:

```python
# plugins/sample_plugin.py

def register(app):
      # app is the Bridge instance
      # Example: mark plugin loaded and print startup message
      print("[PLUGIN] sample_plugin loaded")

      # You can attach custom helper attributes if needed
      app.sample_plugin_enabled = True

      return {
            "name": "sample_plugin",
            "info": "Sample plugin loaded successfully"
      }
```

How to use this sample:
1. Save file as `plugins/sample_plugin.py`.
2. Open app.
3. Open Plugins modal.
4. Click `Reload Plugins`.
5. Confirm plugin appears in list as loaded.


## 5.13.9 Export Modal - Input-by-Input Guide

Input:
- Export format selector (txt/pdf/docx/csv/xlsx)

Steps:
1. Open export action from chat or message.
2. Choose format.
3. Save.

Format recommendation:
- TXT: quick plain text sharing
- PDF: formal report
- DOCX: editable report
- CSV/XLSX: tabular analysis or spreadsheet post-processing


## 5.13.10 Practical End-to-End Usage Scenarios

### Scenario A: Reconciliation in Agent Mode
1. Load model.
2. Agent -> `+ New` instruction.
3. Write reconciliation instruction with explicit checks.
4. Save and open instruction chat.
5. Upload billing and sales files.
6. Send optional note: "Prioritize amount mismatch first."
7. Receive response and output file.
8. Open output from chat or Output Files section.
9. Delete obsolete files from Output Files list.


### Scenario B: Domain QA with RAG in Chat Mode
1. Create RAG DB from knowledge folder.
2. Select RAG DB in sidebar.
3. Ask question using chat input.
4. Validate response with source documents.


### Scenario C: Model Quality Comparison
1. Download two model variants.
2. Load compare modal.
3. Run same prompt across both.
4. Decide preferred model for production use.



## 6. Backend API Surface (Bridge) - Practical Usage

Common methods used from frontend:

### App/System
- `get_app_info()` -> runtime summary (RAM/GPU/theme/mode/model)
- `start_monitor()` -> starts periodic CPU/RAM events

### Models
- `get_models()`
- `select_model(label)`
- `load_model()` / `unload_model()`
- `get_model_status()`
- `get_per_model_settings()` / `save_per_model_settings(...)`

### Chats
- `new_chat()`
- `get_chats()`
- `load_chat(chat_id)`
- `rename_chat(...)`
- `delete_chat(chat_id)`
- `branch_chat(...)`

### Generation
- `send_message(text)`
- `stop_generation()`
- `edit_user_message(...)`

### Settings/Theme
- `get_app_settings()`
- `save_app_settings(settings)`
- `set_theme(theme)`
- `get_system_prompt()` / `set_system_prompt(prompt)`

### RAG
- `get_rag_databases()`
- `select_rag(name)`
- `create_rag_from_folder(...)`
- `create_rag_from_url(...)`
- `delete_rag(name)`
- `reindex_rag(name)`

### Agent Processing
- `get_instruction_templates()`
- `save_instruction_template(name, instructions)`
- `delete_instruction_template(name)`
- `agent_chat(text)`
- `process_files_with_ai(files, instructions, output_format)`
- `list_processed_files()`
- `open_file_location(path)`
- `delete_processed_file(path)`


## 7. Storage Details

### 7.1 SQLite (`chats.db`)
Stores:
- chats and messages
- per-chat metadata (`system_prompt`, selected RAG)
- global key-value settings (`app_settings`, model configs)

### 7.2 Theme Persistence
- Theme is saved in backend settings storage
- Restored during app startup
- Applied to frontend by `data-theme` attribute

### 7.3 Agent Outputs
- Generated in `processed_files/`
- Naming format: `analysis_<timestamp>.<ext>`


## 8. Build and Packaging

Current assets:
- Existing spec: `SIMPLE_AI.spec`
- Existing build script: `build.bat`

A dedicated build script for `agent_web.py` is provided:
- `build_agent_web.bat`

It builds an onedir executable and copies required runtime folders.


## 9. New Build Script - How to Use

1. Open terminal in project root
2. Run:
   - `build_agent_web.bat`
3. On success executable is created in:
   - `dist\SIMPLE_AI_WEB\SIMPLE_AI_WEB.exe`

Script behavior:
- Uses venv python if available
- Installs/updates `pyinstaller`
- Cleans previous build output
- Builds from `agent_web.py`
- Includes `web/` as data
- Copies optional runtime folders (`models`, `rag_databases`, `plugins`, `processed_files`, `exports`, `saved_chats`)


## 10. Troubleshooting Guide

### App opens but old UI appears
- Close app fully
- Delete `webview_data` cache
- Reopen app

### Model not loading
- Verify `.gguf` file exists in `models/`
- Check RAM/VRAM availability
- Try smaller quantized model

### Output files section shows error
- Ensure backend has `list_processed_files()` method (updated build)
- Restart app after updates

### Theme not persistent
- Save from Settings modal
- Ensure DB write permissions to project folder

### Agent returns code instead of results
- Use updated prompt instructions
- Keep instruction explicit: request final table/results, no code


## 11. Operational Best Practices

- Keep one model loaded at a time to save RAM
- Use per-model settings for heavy vs lightweight models
- Keep RAG databases focused by domain (one DB per topic)
- Delete old `processed_files` regularly
- Back up `chats.db` if chat history is important


## 12. Quick Start (Recommended)

1. Place GGUF model in `models/`
2. Launch app via `agent_web.py`
3. Load model
4. Create a new chat and test prompt
5. (Optional) create/select RAG database
6. Open Agent mode and create instruction template
7. Upload files and generate result
8. Manage outputs in Output Files section


## 13. Security and Safety Notes

- `delete_processed_file` only allows files under `processed_files/`
- Local files may contain sensitive data; keep machine secure
- Web search fetches external pages; validate important facts


## 14. Important Files Reference

- Entry: `agent_web.py`
- Core backend: `bridge.py`
- Frontend logic: `web/app.js`
- Frontend layout: `web/index.html`
- Styling/theme/layout: `web/style.css`
- DB layer: `database.py`
- RAG core: `rag_manager.py`
- Plugin loader: `plugin_manager.py`
- Packaging spec: `SIMPLE_AI.spec`
- Standard build script: `build.bat`
- New PyInstaller script: `build_agent_web.bat`


## 15. Versioning Note
This document reflects the current workspace state as of generation time and includes recent updates:
- Theme persistence across app restarts
- Agent output files list with open/delete
- Agent processing prompt tuned for actual analysis results
- Header status pill visual consistency updates
