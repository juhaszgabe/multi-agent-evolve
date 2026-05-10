"""Req — SkillLibrary and ErrorCatalog: add/retrieve with mocked ChromaDB."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_mock_embedder():
    embedder = MagicMock()
    embedder.encode.return_value = np.zeros(384)
    return embedder


def _make_mock_collection(count=0, query_result=None):
    col = MagicMock()
    col.count.return_value = count
    if query_result is not None:
        col.query.return_value = query_result
    return col


def _make_mock_client(collection):
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


# ---------------------------------------------------------------------------
# SkillLibrary
# ---------------------------------------------------------------------------

@patch("chromadb.PersistentClient")
@patch("sentence_transformers.SentenceTransformer")
def test_skill_add_calls_upsert(mock_st, mock_client_cls):
    from memory.skill_library import SkillLibrary

    col = _make_mock_collection()
    mock_client_cls.return_value = _make_mock_client(col)
    mock_st.return_value = _make_mock_embedder()

    lib = SkillLibrary(persist_dir="/fake")
    lib.add(task="count rows", code="df.shape[0]", reward=0.95, workflow_id="wf-001")

    col.upsert.assert_called_once()
    call_kwargs = col.upsert.call_args
    assert call_kwargs.kwargs["ids"] == ["wf-001"]
    assert "df.shape[0]" in call_kwargs.kwargs["documents"][0]


@patch("chromadb.PersistentClient")
@patch("sentence_transformers.SentenceTransformer")
def test_skill_retrieve_empty_collection(mock_st, mock_client_cls):
    from memory.skill_library import SkillLibrary

    col = _make_mock_collection(count=0)
    mock_client_cls.return_value = _make_mock_client(col)
    mock_st.return_value = _make_mock_embedder()

    lib = SkillLibrary(persist_dir="/fake")
    result = lib.retrieve("some task")

    assert result == []
    col.query.assert_not_called()


@patch("chromadb.PersistentClient")
@patch("sentence_transformers.SentenceTransformer")
def test_skill_retrieve_returns_structured_list(mock_st, mock_client_cls):
    from memory.skill_library import SkillLibrary

    query_result = {
        "documents": [["code_a", "code_b"]],
        "metadatas": [[
            {"task": "task_a", "reward": 0.9},
            {"task": "task_b", "reward": 0.85},
        ]],
    }
    col = _make_mock_collection(count=2, query_result=query_result)
    mock_client_cls.return_value = _make_mock_client(col)
    mock_st.return_value = _make_mock_embedder()

    lib = SkillLibrary(persist_dir="/fake", top_k=3)
    result = lib.retrieve("similar task")

    assert len(result) == 2
    assert result[0]["code"] == "code_a"
    assert result[0]["task"] == "task_a"
    assert "reward" in result[0]


# ---------------------------------------------------------------------------
# ErrorCatalog
# ---------------------------------------------------------------------------

@patch("chromadb.PersistentClient")
@patch("sentence_transformers.SentenceTransformer")
def test_error_catalog_add_and_retrieve(mock_st, mock_client_cls):
    from memory.error_catalog import ErrorCatalog

    query_result = {
        "documents": [["bad_code\n---\ndropna()"]],
        "metadatas": [[{
            "task": "compute mean",
            "bad_code": "df['x'].mean()",
            "error_description": "NaN values caused wrong result",
            "fix": "dropna()",
        }]],
    }
    col = _make_mock_collection(count=1, query_result=query_result)
    mock_client_cls.return_value = _make_mock_client(col)
    mock_st.return_value = _make_mock_embedder()

    cat = ErrorCatalog(persist_dir="/fake")
    cat.add(
        task="compute mean",
        bad_code="df['x'].mean()",
        error_description="NaN values caused wrong result",
        fix="dropna()",
        workflow_id="wf-1",
        step_id="s-1",
    )

    upsert_call = col.upsert.call_args
    assert upsert_call.kwargs["ids"] == ["wf-1_s-1"]

    result = cat.retrieve("compute average")
    assert len(result) == 1
    assert result[0]["error_description"] == "NaN values caused wrong result"
    assert result[0]["fix"] == "dropna()"
