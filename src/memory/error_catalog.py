from __future__ import annotations


class ErrorCatalog:
    """
    Stores (task, bad_code, error_description, fix) tuples in ChromaDB.
    Retrieved as proactive warnings before each DataAnalyst call.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 2,
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
            .get_or_create_collection("errors")
        )
        self._embedder = SentenceTransformer(embedding_model)
        self._top_k = top_k

    def add(
        self,
        task: str,
        bad_code: str,
        error_description: str,
        fix: str,
        workflow_id: str,
        step_id: str,
    ) -> None:
        emb = self._embedder.encode(task).tolist()
        doc_id = f"{workflow_id}_{step_id}"
        self._col.upsert(
            ids=[doc_id],
            embeddings=[emb],
            documents=[f"{bad_code}\n---\n{fix}"],
            metadatas=[{
                "task": task,
                "bad_code": bad_code,
                "error_description": error_description,
                "fix": fix,
            }],
        )

    def retrieve(self, task: str) -> list[dict]:
        if self._col.count() == 0:
            return []
        emb = self._embedder.encode(task).tolist()
        n = min(self._top_k, self._col.count())
        res = self._col.query(query_embeddings=[emb], n_results=n)
        return [
            {
                "task": m["task"],
                "bad_code": m["bad_code"],
                "error_description": m["error_description"],
                "fix": m["fix"],
            }
            for m in res["metadatas"][0]
        ]
