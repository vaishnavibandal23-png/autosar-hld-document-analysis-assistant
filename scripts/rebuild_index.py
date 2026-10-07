from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_pipeline import RAGEngine

if __name__ == "__main__":
    result = RAGEngine().build_index()
    print(f"Indexed {result['chunks']} chunks from {result['documents']} documents.")
