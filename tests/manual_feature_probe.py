import json
import sys
import tempfile
from pathlib import Path

import fitz
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from rag_handler import RAGHandlerMixin
from rag_manager import RAGManager
from ui_components import UIComponentsMixin

try:
    from llama_cpp import Llama
except ImportError as exc:
    raise SystemExit(f"llama_cpp import failed: {exc}")


MODELS_DIR = WORKSPACE / "models"


def choose_model_path() -> Path:
    ggufs = sorted(MODELS_DIR.glob("*.gguf"))
    if not ggufs:
        raise SystemExit("No GGUF models found in models/")
    for model_path in ggufs:
        if "gemma" in model_path.name.lower() or "-it" in model_path.name.lower():
            return model_path
    return ggufs[0]


def complete(model, prompt: str, max_tokens: int = 96, temperature: float = 0.1) -> str:
    response = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.8,
        repeat_penalty=1.1,
        stream=False,
        stop=["User:", "Assistant:", "Question:"],
    )
    return response["choices"][0]["text"].strip()


def search_web(query: str, num_results: int = 3) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://html.duckduckgo.com/html/?q={query}"
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for div in soup.find_all("div", class_="result__body")[:num_results]:
        title_tag = div.find("a", class_="result__a")
        snippet_tag = div.find(class_="result__snippet")
        title = title_tag.get_text(strip=True) if title_tag else ""
        link = title_tag.get("href", "") if title_tag else ""
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        if title or snippet:
            results.append(f"- {title} ({link}): {snippet[:200]}")
    return "\n".join(results) if results else "No results found."


class UploadHarness(RAGHandlerMixin):
    def __init__(self, model, uploaded_content: str, rag_manager: RAGManager):
        self.model = model
        self.uploaded_content = uploaded_content
        self.rag_manager = rag_manager
        self.temp_rag_db_name = None
        self.status_updates = []

    def update_status(self, text):
        self.status_updates.append(text)

    def add_message(self, role, text):
        return None

class SearchHarness(RAGHandlerMixin):
    def __init__(self):
        self.status_updates = []

    def update_status(self, text):
        self.status_updates.append(text)

    def add_message(self, role, text):
        return None


class VisionHarness(UIComponentsMixin):
    def __init__(self, model, model_path: str):
        self.model = model
        self.model_path = model_path


def main():
    model_path = choose_model_path()
    model = Llama(
        model_path=str(model_path),
        n_ctx=1024,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False,
    )

    results = {
        "model_used": model_path.name,
    }

    normal_prompt = (
        "You are a concise assistant.\n"
        "User: What is the capital of France? Answer in one short sentence.\n"
        "Assistant:"
    )
    normal_response = complete(model, normal_prompt, max_tokens=48)
    results["normal_chat"] = {
        "response": normal_response,
        "looks_correct": "paris" in normal_response.lower(),
    }

    search_harness = SearchHarness()
    web_results = search_harness.search_web("When was Python 2.0 released?")
    web_prompt = (
        "Use ONLY the web results below.\n\n"
        f"Web Results:\n{web_results}\n\n"
        "Rules:\n"
        "- Copy exact dates, names, and numbers from the web results or excerpts when possible.\n"
        "- Do not infer missing words from truncated snippets.\n"
        "- If the results are insufficient, say so explicitly.\n\n"
        "Question: When was Python 2.0 released? Answer in one short sentence and cite the source title if possible.\n"
        "Answer:"
    )
    web_response = complete(model, web_prompt, max_tokens=80, temperature=0.05)
    results["web_chat"] = {
        "web_results": web_results,
        "response": web_response,
        "mentions_expected_fact": "october 2000" in web_response.lower() or "2000" in web_response.lower(),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)

        pdf_path = base / "sample.pdf"
        pdf_doc = fitz.open()
        pdf_page = pdf_doc.new_page()
        pdf_page.insert_text(
            (72, 72),
            "Apex Labs onboarding guide. Start date: 14 July 2027. Manager: Maya Chen. Location: Chennai.",
        )
        pdf_doc.save(str(pdf_path))
        pdf_doc.close()

        pdf_rag_manager = RAGManager(base_directory=str(base / "pdf_rag_store"))
        pdf_text = pdf_rag_manager._extract_text_from_file(str(pdf_path))
        upload_harness = UploadHarness(model, pdf_text, pdf_rag_manager)
        upload_harness._create_temporary_rag_for_uploaded_file(pdf_path.name)
        pdf_chunk = upload_harness.get_relevant_chunk("What is the start date and manager?")
        pdf_prompt = (
            "Use only the uploaded PDF content below.\n\n"
            f"PDF Content:\n{pdf_chunk}\n\n"
            "Question: What is the start date and manager?\n"
            "Answer:"
        )
        pdf_response = complete(model, pdf_prompt, max_tokens=72, temperature=0.05)
        results["pdf_chat"] = {
            "extracted_text_found": "Maya Chen" in pdf_text and "14 July 2027" in pdf_text,
            "retrieved_chunk": pdf_chunk,
            "response": pdf_response,
            "mentions_expected_fact": "maya" in pdf_response.lower() and ("14 july 2027" in pdf_response.lower() or "2027" in pdf_response.lower()),
        }

        image_path = base / "sample.png"
        image = Image.new("RGB", (240, 120), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 40), "HELLO IMAGE", fill="black")
        image.save(image_path)

        vision_harness = VisionHarness(model, str(model_path))
        image_result = vision_harness._try_multimodal_generate("Read the text in the image.", str(image_path))
        ocr_text = vision_harness._extract_text_from_image(str(image_path))
        image_response = ""
        if ocr_text:
            image_response = complete(
                model,
                "The image was converted to text with OCR. Use only that OCR text to answer.\n\n"
                f"OCR Text:\n{ocr_text}\n\nQuestion: What text appears in the image?\nAnswer:",
                max_tokens=48,
                temperature=0.05,
            )
        results["image_chat"] = {
            "mmproj_present": any("mmproj" in p.name.lower() or "clip" in p.name.lower() for p in MODELS_DIR.iterdir()),
            "multimodal_result": image_result,
            "supported_now": image_result is not None,
            "ocr_text": ocr_text,
            "ocr_response": image_response,
        }

        rag_docs = base / "rag_docs"
        rag_docs.mkdir(exist_ok=True)
        (rag_docs / "orion.txt").write_text(
            "Project Orion launch date is 14 July 2027. The project owner is Maya Chen. Budget is 12 crore INR.",
            encoding="utf-8",
        )
        rag_manager = RAGManager(base_directory=str(base / "rag_store"))
        rag_manager.create_from_folder(str(rag_docs), "smoke_rag", chunk_size=220, chunk_overlap=20)
        rag_results = rag_manager.retrieve("smoke_rag", "What is the Project Orion launch date and who owns it?", k=3)
        rag_context = "\n".join(chunk for chunk, _score in rag_results)
        rag_prompt = (
            "Use only the RAG context below.\n\n"
            f"RAG Context:\n{rag_context}\n\n"
            "Question: What is the Project Orion launch date and who owns it?\n"
            "Answer:"
        )
        rag_response = complete(model, rag_prompt, max_tokens=72, temperature=0.05)
        results["rag_chat"] = {
            "retrieved_context": rag_context,
            "response": rag_response,
            "mentions_expected_fact": "maya" in rag_response.lower() and ("14 july 2027" in rag_response.lower() or "2027" in rag_response.lower()),
        }

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()