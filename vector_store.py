import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, index_dir: Path, model_name: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    @property
    def index_path(self):
        return self.index_dir / "index.faiss"

    @property
    def metadata_path(self):
        return self.index_dir / "metadata.json"

    def build(self, chunks):
        texts = [c["text"] for c in chunks]
        if not texts:
            raise ValueError("No readable text was found in the PDF documents.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))
        self.metadata = chunks
        self.metadata_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        self.index = index

    def load(self):
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return True

    def search(self, query, top_k=5):
        if self.index is None:
            if not self.load():
                raise RuntimeError("Knowledge base has not been built.")

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, ids = self.index.search(vector, min(top_k, len(self.metadata)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results
