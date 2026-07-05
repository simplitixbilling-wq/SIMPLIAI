import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


webview_stub = types.ModuleType("webview")
webview_stub.create_window = mock.MagicMock()
webview_stub.start = mock.MagicMock()
webview_stub.windows = []
sys.modules.setdefault("webview", webview_stub)


from app_core.bridge import Bridge
from app_core.rag_manager import RAGManager

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


def _discover_model_path() -> Path | None:
    models_dir = ROOT / "models"
    ggufs = sorted(models_dir.glob("*.gguf"))
    if not ggufs:
        return None
    for model_path in ggufs:
        name = model_path.name.lower()
        if "gemma" in name or "-it" in name:
            return model_path
    return ggufs[0]


class RoleScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Llama is None:
            raise unittest.SkipTest("llama_cpp is not installed")

        cls.model_path = _discover_model_path()
        if cls.model_path is None:
            raise unittest.SkipTest("No GGUF model found in models/")

        cls.model = Llama(
            model_path=str(cls.model_path),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )
        cls.bridge = object.__new__(Bridge)
        cls.headers = {"User-Agent": "Mozilla/5.0"}

    @classmethod
    def tearDownClass(cls):
        cls.model = None

    @classmethod
    def _prompt(cls, prompt: str, *, max_tokens: int = 140, temperature: float = 0.1) -> str:
        response = cls.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stream=False,
        )
        return response["choices"][0]["text"].strip()

    @classmethod
    def _download_bytes(cls, url: str, *, timeout: int = 60, retries: int = 3) -> bytes:
        last_error = None
        for _ in range(retries):
            try:
                response = requests.get(url, headers=cls.headers, timeout=timeout)
                response.raise_for_status()
                return response.content
            except Exception as exc:
                last_error = exc
        raise AssertionError(f"Failed to download {url}: {last_error}")

    def test_lawyer_web_answers_from_official_context(self):
        cases = [
            {
                "name": "irs_w9",
                "url": "https://www.irs.gov/forms-pubs/about-form-w-9",
                "query": "What does Form W-9 request?",
                "context_terms": ["taxpayer identification number", "certification"],
                "answer_terms": ["taxpayer identification number", "taxpayer's identification number", "tin"],
            },
            {
                "name": "singapore_pdpa",
                "url": "https://www.pdpc.gov.sg/overview-of-pdpa/the-legislation/personal-data-protection-act",
                "query": "What does PDPA stand for?",
                "context_terms": ["personal data protection act"],
                "answer_terms": ["personal data protection act"],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RAGManager(base_directory=str(Path(temp_dir) / "rag_store"))
            for case in cases:
                manager.create_from_url(case["url"], case["name"], chunk_size=500, chunk_overlap=80)
                results = manager.retrieve(case["name"], case["query"], k=3)
                context = "\n".join(chunk for chunk, _score in results).lower()
                self.assertTrue(
                    all(term in context for term in case["context_terms"]),
                    msg=f"Missing expected legal terms in retrieved context for {case['name']}",
                )

                answer = self._prompt(
                    "You are a lawyer. Use only the context below. Answer in one sentence.\n\n"
                    f"Context:\n{context}\n\nQuestion: {case['query']}\nAnswer:",
                    max_tokens=80,
                ).lower()
                self.assertTrue(
                    any(term in answer for term in case["answer_terms"]),
                    msg=f"Answer did not contain expected legal fact for {case['name']}: {answer}",
                )

    def test_chartered_accountant_multi_country_table(self):
        answer = self._prompt(
            "System: You are a Chartered Accountant advising across India, US, UK, Singapore, UAE, and New Zealand. "
            "Return exactly one markdown table with 6 rows and these columns: Country, Main Tax Topic, "
            "Filing/Registration Focus, CA Risk Note. No prose before or after the table.\n"
            "User: Act like a CA and compare indirect-tax/compliance focus across these 6 countries.\n"
            "Assistant:",
            max_tokens=420,
        ).lower()

        for country in ["india", "us", "uk", "singapore", "uae", "new zealand"]:
            self.assertIn(country, answer)
        self.assertIn("| country |", answer)
        self.assertGreaterEqual(len([line for line in answer.splitlines() if line.strip().startswith("|")]), 8)

    def test_finance_reconciliation_and_projection_from_web_data(self):
        csv_text = self._download_bytes(
            "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv",
            timeout=30,
        ).decode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            csv_path = temp_dir / "apple_prices.csv"
            xlsx_path = temp_dir / "apple_prices_compare.xlsx"
            csv_path.write_text(csv_text, encoding="utf-8")

            left = pd.read_csv(csv_path)
            right = left.copy()
            right.loc[0, "AAPL.Close"] = round(float(right.loc[0, "AAPL.Close"]) + 1.23, 2)
            right.to_excel(xlsx_path, index=False)
            right = pd.read_excel(xlsx_path)

            merged = left.merge(right, on="Date", suffixes=("_csv", "_xlsx"))
            diffs = merged[merged["AAPL.Close_csv"].astype(str) != merged["AAPL.Close_xlsx"].astype(str)]
            self.assertEqual(len(merged), 506)
            self.assertEqual(len(diffs), 1)
            self.assertEqual(str(diffs.iloc[0]["Date"]), "2015-02-17")

        html = self._download_bytes(
            "https://www.macrotrends.net/stocks/charts/AAPL/apple/revenue",
            timeout=30,
        ).decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        description = (soup.find("meta", attrs={"name": "description"}) or {}).get("content", "")
        pattern = (
            r"Apple annual revenue for (\d{4}) was <strong>\$([\d.]+)B</strong>, "
            r"a <strong>([\d.]+)% (increase|decline)</strong> from (\d{4})"
        )
        matches = re.findall(pattern, description)
        annual = []
        for year, revenue, pct, direction, _prev_year in matches:
            growth = float(pct)
            if direction == "decline":
                growth = -growth
            annual.append({"Year": int(year), "RevenueB": float(revenue), "GrowthPct": growth})

        annual.sort(key=lambda item: item["Year"])
        self.assertEqual([row["Year"] for row in annual], [2023, 2024, 2025])
        avg_growth = sum(row["GrowthPct"] for row in annual[-3:]) / 3
        self.assertAlmostEqual(avg_growth, 1.8833333333, places=2)

        projections = []
        current = annual[-1]["RevenueB"]
        for step in range(1, 6):
            current = current * (1 + avg_growth / 100.0)
            projections.append(round(current, 2))

        self.assertEqual(len(projections), 5)
        self.assertEqual(projections[0], 424.0)
        self.assertEqual(projections[-1], 456.85)

    def test_finance_pdf_and_scanned_pdf_extraction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            manager = RAGManager(base_directory=str(temp_dir / "rag_store"))

            annual_pdf = temp_dir / "apple_annual_2023.pdf"
            annual_pdf.write_bytes(
                self._download_bytes(
                    "https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AAPL_2023.pdf",
                    timeout=60,
                )
            )
            annual_text = manager._extract_text_from_file(str(annual_pdf)).lower()
            self.assertIn("net sales", annual_text)
            self.assertIn("net income", annual_text)
            self.assertIn("total assets", annual_text)

            html = self._download_bytes(
                "https://www.macrotrends.net/stocks/charts/AAPL/apple/revenue",
                timeout=30,
            ).decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            description = (soup.find("meta", attrs={"name": "description"}) or {}).get("content", "")
            lines = ["Apple Revenue Summary (source: Macrotrends)"]
            for year, revenue, pct, direction, _prev_year in re.findall(
                r"Apple annual revenue for (\d{4}) was <strong>\$([\d.]+)B</strong>, "
                r"a <strong>([\d.]+)% (increase|decline)</strong> from (\d{4})",
                description,
            ):
                lines.append(f"Year {year} Revenue {revenue}B Growth {pct}% {direction}")

            image = Image.new("RGB", (1400, 900), "white")
            draw = ImageDraw.Draw(image)
            y = 60
            for line in lines:
                draw.text((60, y), line, fill="black")
                y += 80

            scanned_pdf = temp_dir / "apple_revenue_scanned.pdf"
            image.save(scanned_pdf, "PDF", resolution=180.0)
            scanned_text = manager._extract_text_from_file(str(scanned_pdf)).lower()
            self.assertIn("2025", scanned_text)
            self.assertIn("416.161", scanned_text)
            self.assertIn("growth", scanned_text)

    def test_finance_invoice_ocr_samples_are_readable(self):
        invoice_names = [
            "AmazonWebServices.pdf",
            "FlipkartInvoice.pdf",
            "oyo.pdf",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            manager = RAGManager(base_directory=str(temp_dir / "rag_store"))
            for invoice_name in invoice_names:
                invoice_path = temp_dir / invoice_name
                invoice_path.write_bytes(
                    self._download_bytes(
                        f"https://raw.githubusercontent.com/invoice-x/invoice2data/master/tests/compare/{invoice_name}",
                        timeout=40,
                    )
                )
                text = manager._extract_text_from_file(str(invoice_path)).lower()
                keyword_hits = sum(
                    1 for token in ("invoice", "date", "total", "amount") if token in text
                )
                self.assertGreater(len(text), 500, msg=f"OCR output too short for {invoice_name}")
                self.assertGreaterEqual(
                    keyword_hits,
                    2,
                    msg=f"Invoice text for {invoice_name} did not expose enough finance fields",
                )

    def test_student_pdf_study_workflow(self):
        pdf_bytes = self._download_bytes("https://ncert.nic.in/textbook/pdf/iesc101.pdf", timeout=60)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            pdf_path = temp_dir / "ncert_science.pdf"
            pdf_path.write_bytes(pdf_bytes)

            manager = RAGManager(base_directory=str(temp_dir / "rag_store"))
            manager.create_from_folder(str(temp_dir), "student_pdf", chunk_size=500, chunk_overlap=80)
            results = manager.retrieve("student_pdf", "What is matter according to the chapter?", k=3)
            context = "\n".join(chunk for chunk, _score in results).lower()
            self.assertIn("matter", context)
            self.assertTrue("occupy space" in context or "mass" in context)

            answer = self._prompt(
                "Use only the PDF context below. Answer in 2 sentences.\n\n"
                f"PDF Context:\n{context}\n\nQuestion: What is matter according to the chapter?\nAnswer:",
                max_tokens=90,
            ).lower()
            self.assertIn("matter", answer)
            self.assertTrue("occupy space" in answer or "mass" in answer)

    def test_student_multilingual_and_essay_support(self):
        web_query = "What is CRISPR-Cas9 used for in biology?"
        web_results = Bridge._search_web(self.bridge, web_query, num_results=3)
        web_answer = self._prompt(
            "Use only the web results below. Give a short research summary in 2 sentences.\n\n"
            f"Web results:\n{web_results}\n\nQuestion: {web_query}\nAnswer:",
            max_tokens=110,
        ).lower()
        self.assertTrue("gene editing" in web_results.lower() or "genome editing" in web_results.lower())
        self.assertTrue("gene editing" in web_answer or "genome editing" in web_answer)

        math_answer = self._prompt(
            "Solve and answer briefly: What is the derivative of x^2 + 3x + 1 with respect to x?",
            max_tokens=40,
        ).lower()
        self.assertTrue("2x + 3" in math_answer or "2x+3" in math_answer)

        science_answer = self._prompt(
            "Answer briefly: Why does ice float on water?",
            max_tokens=70,
        ).lower()
        self.assertIn("ice", science_answer)
        self.assertIn("dense", science_answer)

        hindi = self._prompt("हिंदी में जलवायु परिवर्तन पर 2 वाक्य लिखो।", max_tokens=90, temperature=0.2)
        marathi = self._prompt(
            "मराठीत छत्रपती शिवाजी महाराजांचे महत्त्व दोन वाक्यात सांगा.",
            max_tokens=90,
            temperature=0.2,
        )
        tamil = self._prompt(
            "தமிழில் சங்க இலக்கியம் பற்றி இரண்டு வாக்கியங்கள் எழுதுங்கள்.",
            max_tokens=90,
            temperature=0.2,
        )

        self.assertTrue(any("\u0900" <= char <= "\u097f" for char in hindi))
        self.assertTrue(any("\u0900" <= char <= "\u097f" for char in marathi))
        self.assertTrue(any("\u0b80" <= char <= "\u0bff" for char in tamil))
        self.assertTrue("जलवायु" in hindi or "परिवर्तन" in hindi)
        self.assertTrue("शिवाजी" in marathi or "महाराज" in marathi)
        self.assertTrue("சங்க" in tamil or "இலக்கியம்" in tamil)

        essay = self._prompt(
            "Write a short student essay of about 180 to 220 words on: "
            "The role of artificial intelligence in education.",
            max_tokens=320,
            temperature=0.2,
        )
        word_count = len(essay.split())
        essay_lower = essay.lower()
        self.assertGreaterEqual(word_count, 150)
        self.assertLessEqual(word_count, 260)
        self.assertTrue("artificial intelligence" in essay_lower or "ai" in essay_lower)
        self.assertTrue("education" in essay_lower or "learning" in essay_lower)


if __name__ == "__main__":
    unittest.main(verbosity=2)