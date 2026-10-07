import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()


def build_prompt(question: str, sources: list[dict]) -> str:
    evidence = []
    for i, src in enumerate(sources, 1):
        evidence.append(
            f"[SOURCE {i}] Document: {src['document']} | "
            f"Section: {src.get('section') or 'N/A'} | Page: {src['page']}\n"
            f"{src['text']}"
        )

    context = "\n\n".join(evidence)

    return f"""You are an AUTOSAR HLD document analysis assistant.

Answer the user's question using ONLY the supplied evidence.
Do not invent components, interfaces, ports, signals, dependencies, or relationships.
If the evidence is insufficient, say that the approved HLD evidence is insufficient.
Keep the answer concise and technical.
Cite supporting evidence using [SOURCE 1], [SOURCE 2], etc.

User question:
{question}

Approved evidence:
{context}

Answer:
"""
