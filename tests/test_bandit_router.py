"""LinUCB BanditRouter — deterministic tests, no LLM calls."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from router.actions import RouterAction, TaskState
from router.bandit_router import BanditRouter


def _actions():
    return [
        RouterAction("data_analysis", "python_sandbox"),
        RouterAction("data_analysis", "sql_query"),
        RouterAction("visualization",  "chart_generator"),
    ]


def _nonzero_state():
    s = TaskState()
    s.task_type = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    s.csv_size_bucket = 1
    return s


# ---------------------------------------------------------------------------

def test_select_returns_valid_action():
    router = BanditRouter(_actions(), alpha=0.25)
    chosen = router.select_action(TaskState())
    assert chosen in _actions()


def test_converges_to_rewarded_action():
    actions = _actions()
    router = BanditRouter(actions, alpha=0.1)
    state = _nonzero_state()
    for _ in range(60):
        router.update(state, actions[0], reward=1.0)
        router.update(state, actions[1], reward=0.0)
        router.update(state, actions[2], reward=0.0)
    assert router.select_action(state) == actions[0]


def test_update_invalid_action_silent():
    router = BanditRouter(_actions())
    router.update(TaskState(), RouterAction("data_analysis", "NONEXISTENT"), reward=1.0)
    chosen = router.select_action(TaskState())
    assert chosen in _actions()


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "bandit.npz")
    actions = _actions()
    r1 = BanditRouter(actions, alpha=0.25, state_path=path)
    state = _nonzero_state()
    for _ in range(10):
        r1.update(state, actions[0], reward=1.0)
    selected_r1 = r1.select_action(state)

    r2 = BanditRouter(actions, alpha=0.25, state_path=path)
    assert r2.select_action(state) == selected_r1


def test_feature_vector_dimension():
    vec = TaskState().to_float_array()
    assert len(vec) == 395
    router = BanditRouter(_actions(), state_dim=395)
    chosen = router.select_action(TaskState())
    assert chosen in _actions()
