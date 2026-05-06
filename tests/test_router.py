"""Req 2 — Router abstraction: PlannerRouter, RandomRouter, StaticRouter, BanditRouter stub."""
import pytest
from router import RouterAction, TaskState, PlannerRouter, RandomRouter, StaticRouter, BanditRouter


def _dummy_state() -> TaskState:
    return TaskState()


# ---------------------------------------------------------------------------
# PlannerRouter
# ---------------------------------------------------------------------------

def test_planner_router_reads_plan_step():
    router = PlannerRouter()
    step = {"id": 1, "agent_role": "data_analysis", "suggested_tool": "sql_query",
            "task": "...", "depends_on": []}
    action = router.select_action(_dummy_state(), plan_step=step)
    assert action.agent_role == "data_analysis"
    assert action.tool_name == "sql_query"


def test_planner_router_handles_null_tool():
    router = PlannerRouter()
    step = {"id": 2, "agent_role": "synthesis", "suggested_tool": None,
            "task": "...", "depends_on": []}
    action = router.select_action(_dummy_state(), plan_step=step)
    assert action.agent_role == "synthesis"
    assert action.tool_name is None


def test_planner_router_update_is_noop():
    router = PlannerRouter()
    router.update(_dummy_state(), RouterAction("data_analysis", "python_sandbox"), 0.9)


# ---------------------------------------------------------------------------
# RandomRouter
# ---------------------------------------------------------------------------

def test_random_router_returns_valid_action():
    space = [RouterAction("data_analysis", "python_sandbox"),
             RouterAction("data_analysis", "sql_query")]
    router = RandomRouter(space, seed=42)
    action = router.select_action(_dummy_state())
    assert action in space


def test_random_router_is_seeded_deterministic():
    space = [RouterAction("data_analysis", "python_sandbox"),
             RouterAction("data_analysis", "sql_query"),
             RouterAction("visualization", "chart_generator")]
    r1 = RandomRouter(space, seed=42)
    r2 = RandomRouter(space, seed=42)
    results1 = [r1.select_action(_dummy_state()) for _ in range(10)]
    results2 = [r2.select_action(_dummy_state()) for _ in range(10)]
    assert results1 == results2


def test_random_router_different_seeds_differ():
    space = [RouterAction("data_analysis", "python_sandbox"),
             RouterAction("data_analysis", "sql_query")]
    r1 = RandomRouter(space, seed=1)
    r2 = RandomRouter(space, seed=99)
    results1 = [r1.select_action(_dummy_state()) for _ in range(20)]
    results2 = [r2.select_action(_dummy_state()) for _ in range(20)]
    # With 2 options and 20 draws it's astronomically unlikely they're identical
    assert results1 != results2


# ---------------------------------------------------------------------------
# StaticRouter
# ---------------------------------------------------------------------------

def test_static_router_returns_mapped_action():
    router = StaticRouter({
        "default":     RouterAction("data_analysis", "python_sandbox"),
        "aggregation": RouterAction("data_analysis", "sql_query"),
    })
    action = router.select_action(_dummy_state(), task_category="aggregation")
    assert action.tool_name == "sql_query"


def test_static_router_falls_back_to_default():
    router = StaticRouter({"default": RouterAction("data_analysis", "python_sandbox")})
    action = router.select_action(_dummy_state(), task_category="unknown_category")
    assert action.tool_name == "python_sandbox"


def test_static_router_requires_default_key():
    with pytest.raises(ValueError, match="default"):
        StaticRouter({"aggregation": RouterAction("data_analysis", "sql_query")})


# ---------------------------------------------------------------------------
# BanditRouter stub
# ---------------------------------------------------------------------------

def test_bandit_router_raises_not_implemented():
    router = BanditRouter()
    with pytest.raises(NotImplementedError):
        router.select_action(_dummy_state())


def test_bandit_router_update_raises_not_implemented():
    router = BanditRouter()
    with pytest.raises(NotImplementedError):
        router.update(_dummy_state(), RouterAction("data_analysis", "python_sandbox"), 1.0)
