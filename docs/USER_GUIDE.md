# SIMPLE_AI — Complete User Guide

> **Version:** Latest &nbsp;|&nbsp; **Platform:** Windows &nbsp;|&nbsp; **Engine:** Local GGUF + Ollama

SIMPLE_AI is an offline-first AI chat and agent application. It runs AI models entirely on your machine — no cloud, no subscriptions. This guide covers every feature and how to use it.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Model Management](#2-model-management)
3. [Chat Mode](#3-chat-mode)
4. [File Upload & Attachment](#4-file-upload--attachment)
5. [Export & Download](#5-export--download)
6. [Web Search](#6-web-search)
7. [Knowledge Bases (RAG)](#7-knowledge-bases-rag)
8. [Agent Mode](#8-agent-mode)
9. [Plugins & Tools](#9-plugins--tools)
10. [Voice & Text-to-Speech](#10-voice--text-to-speech)
11. [Image Attachment (Vision)](#11-image-attachment-vision)
12. [Settings & Customization](#12-settings--customization)
13. [Keyboard Shortcuts](#13-keyboard-shortcuts)
14. [System Monitor](#14-system-monitor)
15. [Activation & Trial](#15-activation--trial)

---

## 1. Getting Started

### Launching the App
- Run `agent_web.py` to start the web-based UI.
- The app opens in a native-looking window powered by pywebview.
- The status bar (top-right) shows **"Initializing…"** until ready, then turns **green** when idle.

### First-Time Setup
1. The app auto-detects GGUF model files in the `models/` folder.
2. Select a model from the **model dropdown** in the top bar.
3. Click **Load** to load it into memory.
4. Start chatting once the status shows **"Ready"**.

---

## 2. Model Management

### 2.1 Loading a Model
- **How:** Select a model from the dropdown in the top bar → click the **Load** button (or press `Ctrl+L`).
- The progress bar in the sidebar shows loading progress.
- Once loaded, the sidebar shows the model name, and the status turns green.

### 2.2 Supported Model Types
- **GGUF files** — Local models placed in the `models/` folder. Runs via llama-cpp-python.
- **Ollama models** — If Ollama is running locally, its models also appear in the dropdown.

### 2.3 Downloading Models from HuggingFace
- **How:** Click the **⬇** button in the top bar to open the HuggingFace Model Downloader.
- Type a search query (e.g., `gemma`, `qwen`, `llama`) and click **Search**.
- The app shows recommended GGUF models sized for your system RAM.
- Click a result to start downloading — progress is shown in the modal.
- Downloaded models are saved to the `models/` folder and become immediately available.

### 2.4 Per-Model Settings
- **How:** Click the **⚙** button in the top bar.
- Adjust per-model overrides:
  - **Temperature** — Controls randomness (lower = more focused).
  - **Context window** — Override the default context size (1024–65536 tokens).
  - **CPU threads** — Number of threads for inference.
- Click **Save** to apply, or **Reset** to return to defaults.

### 2.5 Compare Models
- **How:** Go to **Settings → Compare Models**.
- Select two models (Model A and Model B) from the dropdowns.
- Enter a prompt and click **Run Compare**.
- Both models generate side-by-side responses so you can evaluate quality.

### 2.6 Unloading a Model
- Loading a new model automatically unloads the previous one.
- Memory is freed when the model is unloaded.

---

## 3. Chat Mode

### 3.1 Starting a New Chat
- **How:** Click **+ New Chat** in the sidebar (or press `Ctrl+N`).
- Each chat is independent with its own message history and attached files.

### 3.2 Sending a Message
- Type your message in the text box at the bottom and press **Enter** (or click **➤**).
- Use `Shift+Enter` to add a new line without sending.
- The AI generates a response streamed in real-time.

### 3.3 Stopping Generation
- **How:** Click the **■ Stop** button (or press `Escape`) while the AI is generating.
- The response stops immediately and shows what was generated so far.

### 3.4 Reply Length Control
- **How:** Use the **Reply length** dropdown next to the input box.
- Options: Auto, 256, 512, 1024, 2048, 4096, 8192, or 16384 tokens.
- **Auto** adjusts dynamically based on the model's context window.

### 3.5 Message Actions
Every assistant message has an action bar with these options:

| Action | What It Does |
|--------|-------------|
| **Copy** | Copies the response text to your clipboard. |
| **Regenerate** | Deletes the last response and generates a new one. |
| **Branch** | Creates a copy of the chat up to this point, so you can explore a different direction without losing the original. |
| **Speak** | Reads the response aloud using text-to-speech. |
| **Export** | Exports this single message as TXT, PDF, DOCX, CSV, XLSX, or WAV. |

### 3.6 Editing a User Message
- **How:** Double-click any of your sent messages.
- Edit the text in the popup and confirm.
- The conversation is replayed from that point with the edited message.

### 3.7 Managing Chats

| Action | How |
|--------|-----|
| **Search chats** | Type in the search box above the chat list in the sidebar. |
| **Rename a chat** | Right-click a chat in the sidebar → Rename. |
| **Delete a chat** | Right-click a chat in the sidebar → Delete. A confirmation dialog appears. |
| **Switch chats** | Click any chat in the sidebar list. Files and context are restored automatically. |

### 3.8 System Prompt
- **How:** Click the **📝** button in the top bar.
- Enter custom instructions (e.g., "You are a helpful coding assistant specializing in Python").
- The system prompt applies to the current chat only.
- Leave blank to use the default system prompt.

### 3.9 Context Window Display
- **Where:** Sidebar, below the loaded model name.
- Shows: **Context: X/Y (Z%)** — how full the context window is.
- The bar turns **yellow** at 75% and **red** at 90%.
- Prompt token count is shown in the expandable details.

### 3.10 Token Counter
- **Where:** Next to the Send button, shows `X tokens`.
- Estimates how many tokens your current input will use before you send it.

---

## 4. File Upload & Attachment

### 4.1 Upload via Button
- **How:** Click the **📎** button in the input bar.
- A file dialog opens — you can select **one or multiple files**.
- Selecting files via dialog **replaces** any previously attached files.

### 4.2 Drag & Drop Upload
- **How:** Drag files from your desktop/Explorer directly onto the chat area.
- A blue overlay with **"Drop files here"** appears as visual feedback.
- Dropping files **appends** to already attached files (doesn't replace).

### 4.3 File Chips
- After uploading, file names appear as **chips** above the input bar.
- Each chip has an **✕** button to remove that specific file.
- Removing all files clears the attachment.

### 4.4 Supported File Formats

| Category | Extensions |
|----------|-----------|
| Documents | `.txt`, `.md`, `.pdf`, `.docx`, `.rtf` |
| Spreadsheets | `.csv`, `.xlsx`, `.xls`, `.tsv` |
| Data | `.json`, `.xml`, `.yaml`, `.yml` |
| Code | `.py`, `.js`, `.css`, `.sql` |
| Config | `.ini`, `.cfg`, `.toml`, `.log` |
| Web | `.html`, `.htm` |

### 4.5 How File Content Is Used
- **Text files** are read directly and made available to the AI.
- **PDFs** are extracted using PyMuPDF (text + OCR for scanned pages). Large PDFs load in the background.
- **Excel/CSV** are loaded and the AI can analyze their content, answer questions, and generate reports.
- The AI automatically receives relevant file content when you ask questions about the uploaded files.

### 4.6 Multi-File Context
- When multiple files are attached, the AI can see all of them.
- Ask cross-file questions like *"Compare the totals in fileA.csv and fileB.xlsx"*.
- The most recently uploaded file becomes the "primary" file for single-file commands.

### 4.7 Auto-Export Detection
- You can ask the AI to export files in natural language:
  - *"Convert this to Excel"*
  - *"Save as PDF"*
  - *"Turn this into a Word document"*
  - *"Make this a CSV"*
  - *"Export to docx"*
- The app detects these phrases and performs the export directly without LLM intervention.

---

## 5. Export & Download

### 5.1 Export a Single Message
- **How:** Click the **Export** button on any assistant message.
- Choose a format from the modal:

| Format | Output |
|--------|--------|
| **Text (.txt)** | Plain text file |
| **PDF (.pdf)** | Formatted PDF document |
| **Word (.docx)** | Microsoft Word document |
| **CSV (.csv)** | Comma-separated values (for tabular content) |
| **Excel (.xlsx)** | Excel spreadsheet (for tabular content) |
| **WAV (.wav)** | Audio file of the response read aloud |

### 5.2 Export Full Chat
- **How:** Right-click a chat in the sidebar → Export, or use the export option available in message context.
- Exports the entire conversation history.

### 5.3 Natural Language Export
- Simply tell the AI what you want:
  - *"Convert this PDF to Excel"*
  - *"Export the uploaded file as CSV"*
  - *"Turn this into a Word file"*
- The app detects the intent and handles it without needing the AI model.

---

## 6. Web Search

### 6.1 Enabling Web Search
- **How:** Click the **🌐** button in the input bar. It toggles on/off.
- When enabled, the button is highlighted.
- The AI will search the web before answering your question.

### 6.2 How It Works
1. Your question is sent to a search engine.
2. Search results (titles, snippets, URLs) are collected.
3. The AI reads the results and formulates an answer with citations.

### 6.3 Search Providers (Priority Order)
1. **Brave Search API** — Best quality. Requires a free API key (2,000 searches/month free).
2. **DuckDuckGo** — Fallback, no API key needed.
3. **Bing** — Final fallback via HTML scraping.

### 6.4 Setting Up Brave Search (Optional)
- **How:** Go to **Settings → Web Search → Brave Search API Key**.
- Get a free key at [brave.com/search/api](https://brave.com/search/api/).
- Paste it and click **Save**.

### 6.5 Source Cards
- When web search is used, source cards appear showing:
  - Website favicon
  - Page title
  - Domain name
- These help you verify the information sources.

---

## 7. Knowledge Bases (RAG)

RAG (Retrieval-Augmented Generation) lets you create searchable knowledge bases from your documents. The AI retrieves relevant chunks before answering, producing accurate, grounded responses.

### 7.1 Creating a Knowledge Base

**From a folder:**
1. Click the **+** button next to the **KNOWLEDGE** label in the sidebar.
2. Select **Folder** as source type.
3. Enter a **Name** (e.g., "Company Docs").
4. Enter the **Folder path** (e.g., `C:\Documents\contracts`).
5. Adjust **Chunk size** (default 512) and **Overlap** (default 100).
6. Click **Create**.

**From a URL:**
1. Same steps but select **URL** as source type.
2. Enter the web page URL instead of a folder path.
3. The app scrapes the page content and indexes it.

### 7.2 Supported Document Types for RAG
- `.pdf` (including scanned — uses OCR)
- `.txt`, `.md`
- `.docx`
- `.csv`, `.xlsx`, `.xls`
- `.pptx`
- Images (`.jpg`, `.png`, `.webp`, `.bmp`, `.tiff`) — OCR extraction

### 7.3 Using a Knowledge Base in Chat
- **How:** Type `@` followed by the knowledge base name in the chat input.
- An autocomplete dropdown appears showing matching knowledge bases.
- Select one with **Arrow keys + Enter/Tab**, or click it.
- Example: Type `@Invoices` then ask your question.
- The AI will search that knowledge base and use the relevant content to answer.

### 7.4 Auto-Activation
- Click a knowledge base in the sidebar to **select/deselect** it.
- When selected (highlighted), it automatically provides context to every message in that chat without needing `@` mentions.

### 7.5 Retrieval Pipeline
The RAG system uses an 8-signal retrieval pipeline:
1. **TF-IDF similarity** — Semantic-ish matching via bigrams
2. **BM25 scoring** — Classic information retrieval
3. **Keyword index** — Exact word matches
4. **Entity overlap** — Dates, amounts, PAN numbers, percentages
5. **Document-name match** — Boosts chunks from referenced files
6. **Intent detection** — Quantitative, temporal, entity, comparison queries
7. **Structured term matching** — PAN, passport, CIN, reference numbers
8. **Neighbor context boost** — Adjacent chunks for continuity

### 7.6 Managing Knowledge Bases

| Action | How |
|--------|-----|
| **Reindex** | Right-click a knowledge base → Reindex (re-scans the source folder). |
| **Delete** | Right-click a knowledge base → Delete. |
| **Progress** | A progress bar appears at the bottom of the sidebar during indexing. |

---

## 8. Agent Mode

Agent mode is designed for structured, repeatable tasks (data analysis, report generation, reconciliation, etc.) using saved instruction templates.

### 8.1 Switching to Agent Mode
- **How:** Click the **🤖 Agent** button in the top bar.
- The view switches to a split-screen layout:
  - **Left panel:** Saved instructions and output files.
  - **Right panel:** Agent workspace (create/edit/chat).

### 8.2 Creating an Instruction Template
1. Click **+ New** in the instructions panel header.
2. Fill in the form:
   - **Name** — A descriptive name (e.g., "Sales Reconciliation").
   - **Role** — Who the agent is (e.g., "You are a senior data analyst").
   - **Task** — What to accomplish (e.g., "Reconcile these two CSV files").
   - **Steps** — Workflow steps, one per line (e.g., "1. Load both files\n2. Match by Order ID").
   - **Output Format** — Default export: Excel, CSV, PDF, TXT, or UI only.
3. Click **💾 Save Instruction**.

### 8.3 Rewriting Instructions with AI
- **How:** After filling the form, click **✨ Rewrite with AI**.
- The loaded AI model polishes and improves your role, task, and steps.
- Review the changes before saving.

### 8.4 Importing Predefined Templates
- **How:** Click **Import Predefined Set** at the bottom of the instruction form.
- A modal shows available templates (e.g., data analysis, invoice processing).
- Click one to auto-fill the form with a ready-to-use template.

### 8.5 Running an Agent Task
1. Click a saved instruction in the left panel.
2. The agent chat opens in the right panel.
3. Upload your files using the **📎** button in the agent input bar.
4. Type additional context or just click **➤ Send**.
5. The agent processes your files according to the instruction steps.
6. Results appear in the chat and output files appear in the **📁 Output Files** section.

### 8.6 Agent File Upload
- Click **📎** in the agent input bar, or drag files onto the input.
- Supported: `.xlsx`, `.xls`, `.csv`, `.pdf`, `.txt`, `.md`, `.docx`, images.
- Multiple files can be uploaded at once.

### 8.7 Agent Web Search
- **How:** Click the **🌐** button in the agent input bar.
- Works the same as chat mode web search.

### 8.8 Agent URL Scraping
- **How:** Click the **🕷** button in the agent input bar.
- Paste a URL and click to fetch the page data.
- The scraped content becomes available for the agent to analyze.

### 8.9 Output Files
- Processed results are saved to the `processed_files/` folder.
- The **📁 Output Files** section shows all generated files.
- Click a file to open/download it.
- Click **⟳ Refresh** to update the list.

### 8.10 SQL Generation & Execution
- When analyzing tabular data, the agent can generate SQL queries.
- These are executed locally via DuckDB — no external database needed.
- Results are displayed inline and can be exported.

### 8.11 Code Execution
- The agent can write and execute Python code in a sandboxed environment.
- Variables like `UPLOADED_FILE`, `UPLOADED_TEXT`, and `UPLOADED_NAME` are pre-set.
- Libraries available: pandas, numpy, matplotlib, and standard library modules.

### 8.12 Managing Instructions

| Action | How |
|--------|-----|
| **Edit** | Click the ✏️ button on an instruction card. |
| **Delete** | Click the 🗑️ button on an instruction card. Confirmation required. |
| **Resize panels** | Drag the resize handle between left and right panels. |
| **Expand chat** | Click the **⤢** button to make the chat panel full-width. |

---

## 9. Plugins & Tools

### 9.1 Built-In System Tools
The AI can use these tools automatically when needed:

| Command | What It Does |
|---------|-------------|
| `/ls [path]` | List files in a directory. |
| `/tree [path]` | Show directory tree structure. |
| `/find [path] [pattern]` | Find files matching a pattern (e.g., `*.pdf`). |
| `/size [path]` | Show disk space usage. |
| `/info` | Show system information. |
| `/code [python]` | Execute Python code in a sandbox. |

- `/ls`, `/tree`, `/find`, `/size`, `/info` are **safe tools** — they run automatically without permission.
- `/code` and other tools require **user approval** before execution. A permission prompt appears in the chat.

### 9.2 How Tool Calling Works
1. You ask something like *"List files in my Downloads folder"*.
2. The AI recognizes this needs a tool and emits a `<tool_call>` block.
3. The app parses the call, requests permission if needed, executes it, and shows the result.
4. The AI summarizes the result in plain language.

### 9.3 Viewing Installed Plugins
- **How:** Go to **Settings → Plugins**.
- Shows all plugins in the `plugins/` folder with enable/disable toggle.
- Click **Reload** to rescan the plugins folder.

### 9.4 Creating a Plugin with AI
1. Go to **Settings → Plugins → ✨ Create with AI**.
2. Enter a **Plugin name** (used as filename).
3. Describe what the plugin should do in plain English.
4. Click **Generate** — the AI writes the plugin code.
5. Review and edit the generated code.
6. Click **Test** to verify it works.
7. Click **Save & Load** to install it.

### 9.5 Plugin Commands
- Plugins register slash commands (e.g., `/dice`, `/hello`).
- Type `/` in the chat input — an autocomplete dropdown shows all available commands.
- Use **Arrow keys + Enter/Tab** to select a command.

### 9.6 Example Plugins Included
- `dice_roll_plugin.py` — `/dice` command to roll dice.
- `hello_command_plugin.py` — `/hello` greeting command.
- `file_tools_plugin.py` — File utility commands.
- `python_exec_plugin.py` — `/code` Python execution sandbox.
- `ai_code_runner_plugin.py` — AI-assisted code generation and execution.
- `plugins_list_command.py` — `/plugins` to list all loaded plugins.

---

## 10. Voice & Text-to-Speech

### 10.1 Voice Input
- **How:** Click the **◉** button in the input bar.
- Speak your message — it is transcribed and placed in the input box.
- Press Enter to send as usual.

### 10.2 Speaking a Message
- **How:** Click the **Speak** button on any assistant message.
- The response is read aloud using the selected voice.
- Click again to **stop** playback.

### 10.3 Choosing a Voice
- **How:** Go to **Settings → Voice**.
- Select from available system voices in the dropdown.
- The selected voice is used for both Speak and WAV export.

### 10.4 Free Piper Voices
- **How:** Go to **Settings → Free Piper Voices**.
- Browse the catalog of high-quality open-source Piper voices.
- Select one and click **Download**.
- Downloaded voices are saved to `models/voices/piper/`.
- They appear in the Voice dropdown after download.

### 10.5 WAV Export
- **How:** Click **Export** on any assistant message → select **WAV Audio (.wav)**.
- The response is converted to speech and saved as a WAV file.
- Uses the currently selected voice.

---

## 11. Image Attachment (Vision)

### 11.1 Attaching an Image
- **How:** Click the **▦** button in the input bar.
- Select an image file (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`).
- The image is sent to the AI along with your text message.

### 11.2 Vision Support
- Requires a model that supports vision/multimodal inputs (e.g., LLaVA, Gemma with vision).
- The status indicator shows whether the loaded model supports vision.
- Ask questions like *"What's in this image?"* or *"Describe this screenshot"*.

---

## 12. Settings & Customization

### 12.1 Opening Settings
- **How:** Click the **Settings** button at the bottom of the sidebar.

### 12.2 Theme
- **Options:** Dark or Light.
- **How:** Select from the Theme dropdown in Settings → Save.
- The UI switches immediately.

### 12.3 Generation Defaults
| Setting | What It Controls | Range |
|---------|-----------------|-------|
| **Temperature** | Randomness of responses | 0.05 – 1.50 |
| **Top-p** | Nucleus sampling threshold | 0.10 – 1.00 |
| **Repeat penalty** | Penalizes repetitive text | 1.00 – 1.50 |
| **Max response tokens** | Maximum length of AI responses | Auto, 256–16384 |

### 12.4 GPU Information
- Settings shows detected GPU name and VRAM.
- Helps you choose appropriately-sized models.

### 12.5 System Information
- CPU, RAM, and OS details displayed in Settings.

---

## 13. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line (without sending) |
| `Ctrl+N` | New chat |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+L` | Load selected model |
| `Ctrl+Shift+R` | Reload the app |
| `Escape` | Stop generation / close open modal |
| `Arrow Down/Up` | Navigate autocomplete dropdowns |
| `Tab` or `Enter` | Select autocomplete item |

---

## 14. System Monitor

### 14.1 Where
- Bottom of the sidebar: **CPU: X% | RAM: Y%**.
- Updates periodically in the background.

### 14.2 What It Shows
- **CPU usage** — Current processor utilization percentage.
- **RAM usage** — Current memory utilization percentage.
- Helps you monitor system load while running AI models.

---

## 15. Activation & Trial

### 15.1 Trial Period
- The app includes a **30-day free trial** from first launch.
- A **trial pill** in the top bar shows remaining days.

### 15.2 Activation
- After the trial, enter a passkey to unlock full access.
- Passkeys are machine-bound for security.
- Contact the developer for a passkey.

### 15.3 What's Limited During Trial
- All features are available during the trial period.
- After expiry, a passkey is required to continue using the app.

---

## Tips & Tricks

- **@mention + question** is the fastest way to query a knowledge base: `@Invoices what is the total for March?`
- **Drag & drop multiple files** onto the chat for bulk analysis.
- **"Convert this to Excel"** works as a natural language command — no need to use the export button.
- **Branch** a conversation to explore "what if" scenarios without losing the original thread.
- **Double-click** your message to edit it and get a new response from that point.
- Use **Compare Models** to find the best model for your use case before committing.
- Create **reusable Agent instructions** for tasks you repeat often (monthly reports, data reconciliation).
- **Piper voices** are free and sound better than default system voices — download from Settings.
- The **/code** tool can run full Python scripts with pandas, numpy, and matplotlib — ask the AI to generate charts or process data.

---

*Built with ❤️ — Runs 100% locally on your machine.*
