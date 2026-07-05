"""Generation mixin — on_send, _generate, streaming, web search synthesis, image (PySide6)."""

import gc
import os
import re
import threading
import time

import psutil

from app_core.system_tools import TokenOptimizer


class GenerationMixin:
    """Mixin providing all text-generation methods for SimpleAiagentAPP."""

    # ── Thread-safe helpers ────────────────────────────────────────

    def _add_message_sync(self, role, text):
        """Create a message widget on the main thread; block until done."""
        if threading.current_thread() is threading.main_thread():
            self.add_message(role, text)
            return self._last_textbox, self._last_container
        result = {}
        done = threading.Event()

        def _create():
            result["container"] = self.add_message(role, text)
            result["textbox"] = self._last_textbox
            done.set()

        self._run_on_main(_create)
        done.wait(timeout=5)
        return result.get("textbox"), result.get("container")

    def _remove_typing_indicator(self):
        """Thread-safe: destroy typing indicator on main thread."""
        if getattr(self, "typing_indicator", None) is None:
            return
        done = threading.Event()

        def _do():
            try:
                if self.typing_indicator is not None:
                    self.typing_indicator.setParent(None)
                    self.typing_indicator.deleteLater()
            except Exception:
                pass
            self.typing_indicator = None
            done.set()

        self._run_on_main(_do)
        done.wait(timeout=2)

    # ── Public API ─────────────────────────────────────────────────

    def stop_generation(self):
        self.stop_generation_flag = True
        self.update_status("Stopping generation...")

    def on_send(self):
        text = self.input_box.text().strip()

        if not self.current_chat_id:
            self.new_chat()

        model_available = self.model is not None and callable(self.model)

        if not text:
            return

        # Auto-load model if none loaded
        if not model_available and not self.web_search_enabled:
            if hasattr(self, "model_map") and self.model_map:
                best_model = self._auto_select_model()
                if best_model:
                    self.model_path = self.model_map[best_model]
                    self.model_menu.setCurrentText(best_model)
                    self.update_status(f"⏳ Auto-loading {best_model}...")

                    def auto_load_and_send(question):
                        self._load_model_thread()
                        if self.model is not None and callable(self.model):
                            if self.web_search_enabled:
                                self._web_search_and_respond(question)
                            else:
                                self._generate(question)
                        else:
                            self._run_on_main(lambda: self.update_status(
                                "❌ Auto-load failed. Select a model manually."))

                    self.input_box.clear()
                    self.add_message("user", text)
                    self.message_history.append({"role": "user", "content": text})
                    if self.current_chat_id:
                        self.chats[self.current_chat_id] = self.message_history
                    self._auto_save_chat()
                    threading.Thread(target=auto_load_and_send, args=(text,), daemon=True).start()
                    return
                else:
                    self.update_status("Error: No models available. Add .gguf files to models/ folder.")
                    return
            else:
                self.update_status("Error: No models found. Add .gguf files to models/ folder.")
                return

        self.input_box.clear()
        self.add_message("user", text)
        self.message_history.append({"role": "user", "content": text})

        if self.current_chat_id:
            self.chats[self.current_chat_id] = self.message_history
        self._auto_save_chat()

        # Handle image attachment
        _attached_img = self.attached_image
        if _attached_img:
            self.attached_image = None
            self.image_attach_btn.setText("🖼")
            self.image_attach_btn.setStyleSheet("")

        if self.web_search_enabled:
            threading.Thread(target=self._web_search_and_respond, args=(text,), daemon=True).start()
        elif _attached_img:
            threading.Thread(target=self._generate_with_image, args=(text, _attached_img), daemon=True).start()
        else:
            threading.Thread(target=self._generate, args=(text,), daemon=True).start()

    # ── Web search path ────────────────────────────────────────────

    def _web_search_and_respond(self, query):
        self._run_on_main(lambda: self.update_status("🔍 Searching web..."))
        web_results = self.search_web(query)

        if not web_results or len(web_results.strip()) < 10:
            self._run_on_main(lambda: self.add_message(
                "assistant", "🌐 No web results found. Try rephrasing your query."))
            self._run_on_main(lambda: self.update_status("Ready"))
            return

        if self.model:
            self._run_on_main(lambda: self.add_message(
                "assistant", "🤖 Processing web results..."))
            context = f"Web Search Results for '{query}':\n\n{web_results}"
            self._generate_with_context(context, query)
        else:
            clean_results = web_results.replace("**", "").replace("*", "")
            self._run_on_main(lambda r=clean_results: self.add_message(
                "assistant", f"🌐 Web Results:\n\n{r}"))
            self._run_on_main(lambda: self.update_status("Ready"))

    def _generate_with_context(self, context_text, original_query):
        try:
            ctx_window = getattr(self, "actual_n_ctx", 2048)
            max_tokens = min(400, ctx_window // 4)
            max_prompt_chars = (ctx_window - max_tokens - 20) * 4
            trimmed_context = context_text[:max_prompt_chars]

            prompt = f"""You are a web-results assistant.
Use ONLY the provided search snippets below.
Rules:
- Copy exact names, dates, numbers, and short quoted facts from the snippets or excerpts when possible.
- Do not infer missing words from truncated fragments.
- If snippets are insufficient or cut off, say: \"Not enough verified web data in snippets.\"
- Keep answer concise and practical.

Search snippets:
{trimmed_context}

User question: {original_query}

Return format:
Answer: <short answer>
Evidence: <1-2 exact facts copied from snippets>
Sources: <2-3 source titles from snippets>"""

            prompt_tokens = len(prompt.split())
            if prompt_tokens + max_tokens > ctx_window:
                max_tokens = max(32, ctx_window - prompt_tokens - 8)

            response_obj = self.model(
                prompt, max_tokens=max_tokens, temperature=0.1,
                top_p=0.8, stream=False, echo=False
            )

            generated_text = response_obj["choices"][0]["text"].strip()
            generated_text = (generated_text.replace("**", "").replace("__", "")
                              .replace("*", "").replace("#", ""))
            generated_text = re.sub(r"\[[^\]]{20,}\]", "", generated_text).strip()

            if not generated_text or len(generated_text.strip()) < 5:
                generated_text = f"🌐 Web Results:\n\n{trimmed_context}"

            self._run_on_main(lambda t=generated_text: self._update_last_assistant_message(t))
            self.message_history.append({"role": "assistant", "content": generated_text})
            if self.current_chat_id:
                self.chats[self.current_chat_id] = self.message_history
                self._auto_save_chat()
            self._run_on_main(lambda: self.update_status("Ready"))

        except Exception:
            error_msg = f"🌐 Web Results:\n\n{context_text[:2000]}"
            self._run_on_main(lambda t=error_msg: self._update_last_assistant_message(t))
            self._run_on_main(lambda: self.update_status("Ready"))

    def _update_last_assistant_message(self, text):
        try:
            tb = getattr(self, "_last_textbox", None)
            if tb is not None:
                self._set_textbox_text(tb, text)
                self._resize_textbox(tb)
                self._scroll_chat("bottom")
                return
            self.add_message("assistant", text)
        except Exception:
            self.add_message("assistant", text)

    # ── Image generation ───────────────────────────────────────────

    def _generate_with_image(self, text, image_path):
        self.generation_in_progress = True
        self.stop_generation_flag = False
        self._run_on_main(lambda: self.send_btn.setEnabled(False))
        self._run_on_main(lambda: self.stop_btn.setEnabled(True))
        self._run_on_main(lambda: self.update_status("🖼 Processing image..."))

        try:
            if self.model is None:
                self._run_on_main(lambda: self.add_message("assistant", "❌ No model loaded."))
                return

            result = self._try_multimodal_generate(text, image_path)
            if result:
                self._run_on_main(lambda r=result: self.add_message("assistant", r))
                self.message_history.append({"role": "assistant", "content": result})
            else:
                ocr_text = self._extract_text_from_image(image_path)
                if ocr_text:
                    prompt = (
                        "The attached image was converted to text with OCR. "
                        "Use the OCR text below to answer the user's request. "
                        "If the OCR text is insufficient, say so briefly.\n\n"
                        f"OCR Text:\n{ocr_text}\n\n"
                        f"User question: {text}\n\nAnswer:"
                    )
                    ocr_result = self.model(
                        prompt,
                        max_tokens=min(220, max(64, getattr(self, "actual_n_ctx", 2048) // 4)),
                        temperature=0.1,
                        top_p=0.8,
                        stream=False,
                    )["choices"][0]["text"].strip()
                    final_text = (
                        "📷 Image processed with OCR fallback.\n\n"
                        f"{ocr_result}\n\n"
                        f"Extracted text:\n{ocr_text[:1200]}"
                    )
                    self._run_on_main(lambda r=final_text: self.add_message("assistant", r))
                    self.message_history.append({"role": "assistant", "content": final_text})
                else:
                    fname = os.path.basename(image_path)
                    note = (f"📷 Image attached: **{fname}**\n\n"
                            "⚠️ The current model doesn't support vision/multimodal input, "
                            "and no readable OCR text was found in the image. "
                            "To use full image analysis, load a LLaVA or multimodal GGUF model "
                            "and place its mmproj clip file in the models/ folder.\n\n"
                            "Answering your text query instead...")
                    self._run_on_main(lambda n=note: self.add_message("assistant", n))
                    self.message_history.append({"role": "assistant", "content": note})
                    self._generate(text)
                    return

            if self.current_chat_id:
                self.chats[self.current_chat_id] = self.message_history
            self._auto_save_chat()
        except Exception as e:
            self._run_on_main(lambda err=str(e): self.add_message(
                "assistant", f"❌ Image error: {err}"))
        finally:
            self.generation_in_progress = False
            self._run_on_main(lambda: self.send_btn.setEnabled(True))
            self._run_on_main(lambda: self.stop_btn.setEnabled(False))
            self._run_on_main(lambda: self.update_status("Ready"))

    # ── Main generation ────────────────────────────────────────────

    def _generate(self, text):
        self.generation_in_progress = True
        self.stop_generation_flag = False
        self._run_on_main(lambda: self.send_btn.setEnabled(False))
        self._run_on_main(lambda: self.stop_btn.setEnabled(True))
        self._run_on_main(lambda: self.update_status("AI is thinking..."))

        if self.model is None or not callable(self.model):
            self._run_on_main(lambda: self.update_status("Error: No model loaded"))
            self._run_on_main(lambda: self.add_message(
                "assistant",
                "❌ Model not available. Please load a model or enable web search."))
            self.generation_in_progress = False
            self._run_on_main(lambda: self.send_btn.setEnabled(True))
            self._run_on_main(lambda: self.stop_btn.setEnabled(False))
            return

        model_cfg = getattr(self, "model_config", self._get_model_config())

        # Add a lightweight loading bubble while the model starts responding.
        _, self.typing_indicator = self._add_message_sync("loading", "..... loading")

        try:
            # 1. DETECT CONTEXT WINDOW
            model_ctx = None
            if hasattr(self.model, "n_ctx"):
                model_ctx = getattr(self.model, "n_ctx")
            if callable(model_ctx):
                try:
                    model_ctx = model_ctx()
                except Exception:
                    model_ctx = None
            if model_ctx is not None:
                try:
                    model_ctx = int(model_ctx)
                except Exception:
                    model_ctx = None

            actual_context_window = model_ctx if model_ctx and model_ctx > 0 else 512

            # 2. TOKEN BUDGET
            prompt_overhead = 80
            min_response_tokens = 50
            available_for_context = max(32, actual_context_window - prompt_overhead - min_response_tokens)

            if actual_context_window >= 2048:
                max_tokens = min(512, actual_context_window // 4)
            elif actual_context_window >= 1024:
                max_tokens = min(256, actual_context_window // 4)
            else:
                max_tokens = min(150, available_for_context)

            self._run_on_main(lambda: self.update_status(
                f"Model context: {actual_context_window} tokens"))

            available_ram = psutil.virtual_memory().available / (1024 ** 3)
            if available_ram < 2:
                max_tokens = min(max_tokens, 100)
                self._run_on_main(lambda: self.update_status(
                    "⚠️ Low RAM: Limiting response length"))

            # RAG CONTEXT
            clean_text, rag_names = self._extract_rag_references(text)
            rag_context = ""
            rag_sources = []

            if self.current_rag_database and self.current_rag_database not in rag_names:
                rag_names.append(self.current_rag_database)

            if rag_names:
                self._run_on_main(lambda: self.update_status(
                    f"📚 Retrieving from RAG: {', '.join(rag_names)}..."))
                rag_context, rag_sources = self._retrieve_rag_context(rag_names, clean_text, k=5)

            text = clean_text if clean_text else text

            # File context
            context = ""
            file_source = ""
            if hasattr(self, "uploaded_content") and self.uploaded_content:
                context = self.get_relevant_chunk(text)
                file_source = "\n[Using uploaded file data]"

            # Combine all contexts
            all_context = ""
            conv_context = self._get_conversation_context(max_messages=10)
            if conv_context:
                all_context += conv_context + "\n\n"
            if rag_context:
                all_context += rag_context + "\n\n"
            if context:
                all_context += f"--- UPLOADED FILE DATA ---\n{context}\n--- END FILE DATA ---\n\n"

            if hasattr(self, "uploaded_content") and self.uploaded_content and not context and not rag_context:
                context = self.uploaded_content.strip()[:max(200, available_for_context * 4)]

            # CONTEXT WINDOW SAFETY TRIM
            effective_window = model_ctx if model_ctx else model_cfg.get("max_context_tokens", 2048)
            reserved = 256
            allowed_context_tokens = max(128, effective_window - reserved)

            all_tokens = all_context.split()
            if len(all_tokens) > allowed_context_tokens:
                all_tokens = all_tokens[-allowed_context_tokens:]
                all_context = " ".join(all_tokens)
                self._run_on_main(lambda: self.update_status(
                    "⚠️ Context trimmed to fit model window"))

            # BUILD PROMPT
            if all_context:
                if actual_context_window < 1024:
                    prompt = f"""{all_context}

Q: {text}
A:"""
                elif rag_context:
                    prompt = f"""You are a document assistant. Answer the question using ONLY the information in the documents below.
Rules:
- Extract the exact answer from the documents. Quote specific values, dates, names.
- If the answer spans multiple documents, combine the information.
- If the documents do not contain the answer, say "Not found in the provided documents."
- Do NOT add information from your own knowledge.

Documents:
{all_context}

Question: {text}

Answer:"""
                else:
                    prompt = f"""Based on:
{all_context}

Question: {text}

Answer:"""
            else:
                if actual_context_window < 1024:
                    prompt = f"{text}"
                else:
                    prompt = f"""Question: {text}

Answer:"""

            # System prompt
            _sys = self.chat_system_prompts.get(self.current_chat_id or "", "").strip()
            if _sys:
                prompt = f"System: {_sys}\n\n{prompt}"

            # PROMPT GUARD
            prompt_tokens = len(prompt.split())
            effective_window = model_ctx or model_cfg.get("max_context_tokens", 2048)
            if model_ctx is None:
                effective_window = min(effective_window, 2048)

            if prompt_tokens + max_tokens > effective_window:
                adjusted_max = max(16, effective_window - prompt_tokens - 8)
                if adjusted_max < max_tokens:
                    max_tokens = adjusted_max
                    self._run_on_main(lambda: self.update_status(
                        "⚠️ Adjusted max_tokens to avoid context overflow"))

            if prompt_tokens + max_tokens > effective_window:
                max_tokens = max(16, effective_window - prompt_tokens - 8)
            if max_tokens < 16:
                max_tokens = 16

            _ew = int(effective_window)
            _pt = int(prompt_tokens + max_tokens)
            self._run_on_main(lambda u=_pt, t=_ew: self._update_ctx_bar(u, t))

            # ── Remove typing indicator and create streaming bubble ──
            self._remove_typing_indicator()
            textbox, _ = self._add_message_sync("assistant", "")

            response = ""
            generated_tokens = 0

            # ── Streaming ──
            temp = model_cfg.get("temperature", 0.25)
            if all_context:
                temp = 0.15

            stream_success = False

            try:
                stream = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=self.app_settings.get("top_p", model_cfg.get("top_p", 0.9)),
                    repeat_penalty=self.app_settings.get("repeat_penalty", 1.1),
                    stream=True,
                )

                corruption_count = 0
                _in_think_block = False
                _think_buffer = ""
                _display_response = ""

                for chunk in stream:
                    if self.stop_generation_flag:
                        self._run_on_main(lambda: self.update_status(
                            "Generation stopped by user"))
                        break

                    try:
                        if not isinstance(chunk, dict):
                            continue
                        if "choices" not in chunk or not chunk["choices"]:
                            continue
                        if "text" not in chunk["choices"][0]:
                            continue

                        token = chunk["choices"][0]["text"]
                        if not token:
                            continue

                        try:
                            token.encode("utf-8").decode("utf-8")
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            corruption_count += 1
                            if corruption_count > 5:
                                self._run_on_main(lambda: self.update_status(
                                    "Warning: Stream quality degraded"))
                                break
                            continue

                        # THINK TAG SUPPRESSION
                        response += token
                        _think_buffer += token

                        if "<think>" in _think_buffer.lower():
                            _in_think_block = True
                            pre = _think_buffer.lower().split("<think>")[0]
                            if pre.strip():
                                _display_response += pre
                            _think_buffer = _think_buffer[
                                _think_buffer.lower().index("<think>") + 7:]

                        if _in_think_block:
                            if "</think>" in _think_buffer.lower():
                                post = _think_buffer[
                                    _think_buffer.lower().index("</think>") + 8:]
                                _think_buffer = post
                                _in_think_block = False
                                if post.strip():
                                    _display_response += post
                                    _think_buffer = ""
                            generated_tokens += 1
                            if generated_tokens % 15 == 0:
                                self._run_on_main(lambda: self.update_status(
                                    "🧠 Reasoning..."))
                            continue

                        if _think_buffer:
                            _display_response += _think_buffer
                            _think_buffer = ""

                        generated_tokens += 1

                        if generated_tokens % 4 == 0:
                            try:
                                safe_text = _display_response.encode(
                                    "utf-8", errors="ignore").decode("utf-8")
                                self._run_on_main(
                                    lambda r=safe_text, tb=textbox:
                                        self._set_textbox_text(tb, r))
                            except Exception:
                                pass

                        if generated_tokens % 12 == 0:
                            self._run_on_main(
                                lambda: self._scroll_chat("bottom"))

                        if generated_tokens % 8 == 0:
                            percent = min(
                                int((generated_tokens / max_tokens) * 100), 100)
                            self._run_on_main(
                                lambda p=percent, t=generated_tokens:
                                    self.update_status(
                                        f"Thinking... {p}% ({t} tokens)"))

                    except Exception:
                        continue

                if response and len(response.strip()) >= 10:
                    stream_success = True

            except Exception as stream_err:
                print(f"[STREAM ERROR] {stream_err}")

            # FALLBACK: non-streaming
            if not stream_success or (response and len(response.strip()) < 10):
                try:
                    self._run_on_main(lambda: self.update_status(
                        "Using direct response mode..."))
                    simple_prompt = prompt.replace(
                        "IMPORTANT: Base your answer ONLY on the following provided content",
                        "Answer directly based on the content",
                    )
                    response_obj = self.model(
                        simple_prompt,
                        max_tokens=min(200, max_tokens),
                        temperature=0.1,
                        stream=False,
                    )
                    if response_obj and "choices" in response_obj:
                        response = response_obj["choices"][0]["text"].strip()
                        if response:
                            stream_success = True
                except Exception as fallback_err:
                    print(f"[FALLBACK ERROR] {fallback_err}")
                    response = ("⚠️ I had trouble processing your question. "
                                "Please try asking about specific details from the document.")

            if not response or len(response.strip()) < 5:
                response = ("⚠️ The response was incomplete. "
                            "Please try rephrasing your question.")

            # DETECT INCOMPLETE
            incomplete = False
            min_response_length = 50
            if len(response.strip()) > min_response_length:
                if response.strip()[-1] not in [".", "!", "?", ":", "`"]:
                    incomplete = True
            elif len(response.strip()) > 0:
                incomplete = True

            # AUTO CONTINUE (disabled)
            attempts = 0
            max_continuation_attempts = 0

            while (incomplete and attempts < max_continuation_attempts
                   and not self.stop_generation_flag):
                attempts += 1
                continuation_prompt = (
                    "Continue and complete the following. "
                    "Do not repeat existing text and be concise:\n\n"
                    f"{response}\n\nContinue:"
                )
                continuation_prompt_tokens = len(continuation_prompt.split())
                continuation_tokens = min(300, self.config["max_tokens"] // 3)
                if continuation_prompt_tokens + continuation_tokens > effective_window:
                    continuation_tokens = max(
                        8, effective_window - continuation_prompt_tokens - 4)
                if continuation_tokens < 8:
                    continuation_tokens = 8

                stream2 = self.model(
                    continuation_prompt, max_tokens=continuation_tokens, stream=True)
                continuation_response = ""
                for chunk in stream2:
                    if self.stop_generation_flag:
                        self._run_on_main(lambda: self.update_status(
                            "Generation stopped by user"))
                        break
                    token = chunk["choices"][0]["text"]
                    response += token
                    continuation_response += token
                    self._run_on_main(
                        lambda r=response, tb=textbox:
                            self._set_textbox_text(tb, r))
                    time.sleep(0.01)

                if len(continuation_response.strip()) > 15:
                    if response.strip()[-1] in [".", "!", "?", ":", "`"]:
                        incomplete = False
                else:
                    incomplete = False

            # ── Cleanup & Format ──
            response = response.strip()
            response = re.sub(
                r"<think>.*?</think>", "", response,
                flags=re.DOTALL | re.IGNORECASE).strip()
            response = re.sub(
                r"<think>.*", "", response,
                flags=re.DOTALL | re.IGNORECASE).strip()

            lines = response.split("\n")
            cleaned_lines = []
            skip_phrases = [
                "search for", "here's how", "visit",
                "check out", "try searching", "follow these",
            ]
            for line in lines:
                line_lower = line.lower()
                if any(p in line_lower for p in skip_phrases):
                    if "search" in line_lower or "visit" in line_lower:
                        continue
                cleaned_lines.append(line)
            response = "\n".join(cleaned_lines).strip()

            max_response_length = min(2000, actual_context_window * 3)
            if len(response) > max_response_length:
                response = response[:max_response_length].rstrip()
                last_period = response.rfind(".")
                if last_period > max_response_length * 0.6:
                    response = response[: last_period + 1]

            try:
                response = response.encode("utf-8", errors="ignore").decode("utf-8")
            except Exception:
                pass

            self._run_on_main(
                lambda r=response, tb=textbox: self._set_textbox_text(tb, r))

            # Hallucination check
            if all_context and response.strip():
                context_words = set(all_context.lower().split())
                response_words = set(response.lower().split())
                overlap = len(context_words & response_words)
                if len(context_words) > 0:
                    overlap_ratio = overlap / len(context_words)
                    if overlap_ratio < 0.02 and len(response.split()) > 40:
                        response = (response.rstrip() +
                                    "\n\n⚠️ Note: This response may not be "
                                    "grounded in the provided sources.")

            # RAG citation
            if rag_sources:
                response = (response.rstrip() +
                            "\n\n📄 Sources: " + ", ".join(rag_sources))

            # Final render with markdown
            self._run_on_main(
                lambda r=response, tb=textbox: self._render_markdown(tb, r))
            self._after(80,
                lambda tb=textbox: self._resize_textbox(tb))

            # Save
            self.message_history.append({"role": "assistant", "content": response})
            if self.current_chat_id:
                self.chats[self.current_chat_id] = self.message_history
            self._auto_save_chat()

        except Exception as e:
            self._run_on_main(lambda err=str(e): self.add_message("assistant", err))

        finally:
            self.generation_in_progress = False
            self.stop_generation_flag = False
            self._run_on_main(lambda: self.send_btn.setEnabled(True))
            self._run_on_main(lambda: self.stop_btn.setEnabled(False))
            self._run_on_main(lambda: self.update_status("Ready"))

    # ── Conversation context helpers ───────────────────────────────

    def _get_conversation_context(self, max_messages=10):
        if not self.message_history or len(self.message_history) < 1:
            return ""

        recent = self.message_history[-max_messages:]
        if not recent:
            return ""

        ctx_window = getattr(self, "actual_n_ctx", 2048)
        max_conv_chars = int(ctx_window * 0.3) * 4

        conversation = ""
        total_chars = 0

        for m in recent:
            role = "User" if m["role"] == "user" else "AI"
            content = m["content"].strip()
            if not content:
                continue
            content = re.sub(
                r"<think>.*?</think>", "", content,
                flags=re.DOTALL | re.IGNORECASE).strip()
            content = re.sub(
                r"<think>.*", "", content,
                flags=re.DOTALL | re.IGNORECASE).strip()
            if not content:
                continue

            max_msg_chars = max_conv_chars // max(len(recent), 1)
            if len(content) > max_msg_chars:
                content = content[:max_msg_chars] + "..."

            line = f"{role}: {content}\n"
            if total_chars + len(line) > max_conv_chars:
                break
            conversation += line
            total_chars += len(line)

        return conversation.strip()

    def _get_smart_context(self, query, max_tokens):
        if not hasattr(self, "uploaded_content") or not self.uploaded_content:
            return ""

        lines = self.uploaded_content.split("\n")
        query_words = set(query.lower().split())
        relevant_lines = []

        for line in lines:
            line_lower = line.lower()
            score = sum(1 for word in query_words if word in line_lower)
            if score > 0 or len(relevant_lines) < 5:
                relevant_lines.append((line, score))

        relevant_lines.sort(key=lambda x: x[1], reverse=True)

        context = ""
        token_count = 0
        token_per_line = 15
        max_lines = max_tokens // token_per_line

        for line, _score in relevant_lines[:max_lines]:
            if line.strip():
                context += line + "\n"
                token_count += token_per_line
                if token_count >= max_tokens:
                    break

        return context[: max_tokens * 4]

    def _chunk_context(self, context, max_tokens):
        if not context:
            return ""
        max_chars = max_tokens * 4
        if len(context) <= max_chars:
            return context
        sections = context.split("\n\n")
        result = ""
        for section in sections:
            if len(result) + len(section) <= max_chars:
                result += section + "\n\n"
            else:
                break
        return result.strip()
