import time
import re

from config import (
    DOCUMENTS_DIR,
    INDEX_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    MIN_RETRIEVAL_SCORE,
)
from document_processor import load_all_documents
from vector_store import VectorStore
from llm import OllamaClient, build_prompt


class RAGEngine:
    def __init__(self):
        self.store = VectorStore(INDEX_DIR, EMBEDDING_MODEL)
        self.llm = OllamaClient(OLLAMA_BASE_URL, OLLAMA_MODEL)

    def build_index(self):
        chunks, pdfs = load_all_documents(DOCUMENTS_DIR)
        self.store.build(chunks)
        return {"documents": len(pdfs), "chunks": len(chunks)}

    def _fallback_answer(self, question, sources):
        if not sources:
            return "No relevant evidence was retrieved from the approved HLD documents."

        # Safe extractive fallback: returns the strongest evidence instead of hallucinating.
        top = sources[:3]
        lines = [
            "The local LLM is unavailable, so the answer below is based directly on the retrieved HLD evidence:"
        ]
        for i, src in enumerate(top, 1):
            snippet = re.sub(r"\s+", " ", src["text"]).strip()
            if len(snippet) > 450:
                snippet = snippet[:450].rsplit(" ", 1)[0] + "..."
            lines.append(
                f"- {snippet} [SOURCE {i}]"
            )
        return "\n".join(lines)

    def _validate(self, answer, sources):
        warnings = []
        if not sources:
            warnings.append("No supporting source was retrieved.")
            return warnings

        cited = re.findall(r"\[SOURCE\s+\d+\]", answer.upper())
        if not cited:
            warnings.append("The generated response did not contain explicit source markers.")

        max_source = len(sources)
        for marker in cited:
            number = int(re.search(r"\d+", marker).group())
            if number < 1 or number > max_source:
                warnings.append("A citation refers to a source outside the retrieved evidence.")

        if sources[0]["score"] < MIN_RETRIEVAL_SCORE:
            warnings.append("Top retrieval similarity is low; verify the answer manually.")

        return sorted(set(warnings))

    def ask(self, question, top_k=5):
        started = time.perf_counter()
        sources = self.store.search(question, top_k=top_k)

        try:
            prompt = build_prompt(question, sources)
            answer = self.llm.generate(prompt)
            if not answer:
                raise RuntimeError("Empty LLM response")
        except Exception:
            answer = self._fallback_answer(question, sources)

        warnings = self._validate(answer, sources)
        latency = time.perf_counter() - started

        return {
            "answer": answer.replace("\n", "<br>"),
            "sources": sources,
            "warnings": warnings,
            "latency": latency
        }
