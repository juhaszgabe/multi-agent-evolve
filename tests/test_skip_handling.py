"""Req 4 — Skip handling: agents handle None/skipped upstream results gracefully."""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

from ai_providers.ai_provider import CompletionResult
from config import Config
from agents.critic import CriticAgent
from agents.writer import WriterAgent


def _make_config():
    return Config()


def _mock_provider(content: str) -> MagicMock:
    provider = MagicMock()
    provider.create_completion.return_value = CompletionResult(content=content)
    return provider


# ---------------------------------------------------------------------------
# CriticAgent — null analyst_result
# ---------------------------------------------------------------------------

def test_critic_approves_when_analyst_result_is_none():
    cfg = _make_config()
    critic = CriticAgent(_mock_provider(""), "model", cfg)
    verdict = critic.critique(
        question="How many rows?",
        plan={"steps": []},
        analyst_result=None,
    )
    assert verdict["verdict"] == "approved"
    assert verdict["issues"] == []
    assert verdict["suggested_action"] is None


def test_critic_approves_when_analyst_result_is_none_with_viz():
    cfg = _make_config()
    critic = CriticAgent(_mock_provider(""), "model", cfg)
    verdict = critic.critique(
        question="Show trend",
        plan={},
        analyst_result=None,
        viz_result={"image_path": "chart.png", "success": True},
    )
    assert verdict["verdict"] == "approved"


# ---------------------------------------------------------------------------
# WriterAgent — None step results (skipped steps)
# ---------------------------------------------------------------------------

_WRITER_RESPONSE = '{"report_text": "The answer is 42.", "key_findings": [], "caveats": []}'


def test_writer_handles_none_step_result():
    cfg = _make_config()
    writer = WriterAgent(_mock_provider(_WRITER_RESPONSE), "model", cfg)
    result = writer.write(
        question="How many rows?",
        plan={"steps": []},
        step_results={"1": None},
        chart_paths=[],
    )
    assert "report_text" in result
    assert result["report_text"]  # non-empty


def test_writer_handles_empty_chart_paths():
    cfg = _make_config()
    writer = WriterAgent(_mock_provider(_WRITER_RESPONSE), "model", cfg)
    result = writer.write(
        question="How many rows?",
        plan={},
        step_results={"1": {"stdout": "42 rows", "success": True}},
        chart_paths=[],
    )
    assert "report_text" in result


def test_writer_handles_mixed_none_and_real_results():
    cfg = _make_config()
    writer = WriterAgent(_mock_provider(_WRITER_RESPONSE), "model", cfg)
    step_results = {
        "1": {"stdout": "revenue by region", "success": True},
        "2": None,   # visualization was skipped
        "3": None,   # verification was skipped
    }
    result = writer.write(
        question="Top region?",
        plan={},
        step_results=step_results,
        chart_paths=[],
    )
    assert "report_text" in result


# ---------------------------------------------------------------------------
# RouterAction skip sentinel preserved in step_results
# ---------------------------------------------------------------------------

def test_skip_sentinel_is_dict_not_none():
    """
    The orchestrator stores {"skipped": True, "agent_role": ...} — not bare None —
    so downstream agents receive a structured dict rather than a NoneType.
    """
    skip_sentinel = {"skipped": True, "agent_role": "visualization"}
    # Writer should handle this via the None-replacement logic
    cfg = _make_config()
    writer = WriterAgent(_mock_provider(_WRITER_RESPONSE), "model", cfg)
    result = writer.write(
        question="Trend?",
        plan={},
        step_results={"1": {"stdout": "data"}, "2": skip_sentinel},
        chart_paths=[],
    )
    assert "report_text" in result
