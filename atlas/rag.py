"""
ATLAS - RAG module
==================
Loads artwork JSON sheets from data/artworks/, embeds them with
sentence-transformers, stores them in ChromaDB (in-memory), and
exposes:
  - search(query, n=1) -> [{"id", "score", "sheet"}, ...]
  - sheet_by_id(artwork_id) -> sheet dict or None
"""

from __future__ import annotations
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTWORKS_DIR = PROJECT_ROOT / "data" / "artworks"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _build_passage(sheet: dict) -> str:
    return (
        f"{sheet.get('title', '')} by {sheet.get('artist', '')}. "
        f"Themes: {', '.join(sheet.get('themes', []))}. "
        f"{sheet.get('long_description', sheet.get('short_description', ''))}"
    )


class RAG:
    def __init__(self, artworks_dir: Path | str = DEFAULT_ARTWORKS_DIR):
        self.artworks_dir = Path(artworks_dir)
        if not self.artworks_dir.is_dir():
            raise RuntimeError(
                f"Artworks directory not found: {self.artworks_dir}. "
                "Create it and add JSON sheets."
            )

        self.sheets: dict[str, dict] = {}
        for path in sorted(self.artworks_dir.glob("*.json")):
            try:
                with open(path) as f:
                    sheet = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON in {path}: {e}")
            sheet_id = sheet.get("id")
            if not sheet_id:
                raise RuntimeError(f"Sheet {path} missing 'id' field.")
            self.sheets[sheet_id] = sheet

        if not self.sheets:
            raise RuntimeError(f"No artwork sheets found in {self.artworks_dir}.")

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        client = chromadb.Client()
        try:
            client.delete_collection("artworks")
        except Exception:
            pass
        self.collection = client.create_collection("artworks")

        ids = list(self.sheets.keys())
        passages = [_build_passage(self.sheets[i]) for i in ids]
        embeddings = self.embedder.encode(passages).tolist()
        self.collection.add(
            ids=ids,
            documents=passages,
            embeddings=embeddings,
        )

    def search(self, query: str, n: int = 1) -> list[dict]:
        q_emb = self.embedder.encode([query]).tolist()
        n = min(n, len(self.sheets))
        results = self.collection.query(query_embeddings=q_emb, n_results=n)
        if not results["ids"][0]:
            return []
        out = []
        for rid, dist in zip(results["ids"][0], results["distances"][0]):
            similarity = 1.0 - dist
            out.append({
                "id": rid,
                "score": similarity,
                "sheet": self.sheets[rid],
            })
        return out

    def sheet_by_id(self, artwork_id: str) -> dict | None:
        return self.sheets.get(artwork_id)


if __name__ == "__main__":
    rag = RAG()
    print(f"Loaded {len(rag.sheets)} sheets:")
    for sid in rag.sheets:
        print(f"  - {sid}")
    print("\nSearch test: 'tell me about the mona lisa'")
    for r in rag.search("tell me about the mona lisa", n=3):
        print(f"  {r['id']:30s} score={r['score']:.3f}")
