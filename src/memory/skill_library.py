from __future__ import annotations


class SkillLibrary:
    """
    Stores successful (task, code) pairs in ChromaDB and retrieves top-k
    most similar examples as few-shot context before each DataAnalyst call.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
    ) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "enable_memory=True requires: pip install '.[memory]'"
            )
        self._col = (
            chromadb.PersistentClient(path=persist_dir)
            .get_or_create_collection("skills")
        )
        self._embedder = SentenceTransformer(embedding_model)
        self._top_k = top_k

    def add(self, task: str, code: str, reward: float, workflow_id: str) -> None:
        emb = self._embedder.encode(task).tolist()
        self._col.upsert(
            ids=[workflow_id],
            embeddings=[emb],
            documents=[code],
            metadatas=[{"task": task, "reward": reward}],
        )

    def retrieve(self, task: str) -> list[dict]:
        if self._col.count() == 0:
            return []
        emb = self._embedder.encode(task).tolist()
        n = min(self._top_k, self._col.count())
        res = self._col.query(query_embeddings=[emb], n_results=n)
        return [
            {"task": m["task"], "code": doc, "reward": m["reward"]}
            for doc, m in zip(res["documents"][0], res["metadatas"][0])
        ]
