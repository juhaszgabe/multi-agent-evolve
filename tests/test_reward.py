"""Req 7 — RewardFunction produces sensible scalar rewards and breakdown."""
import pytest
from logging_schema import StepRecord, WorkflowOutcome
from reward import RewardFunction, MAX_COST_USD, MAX_LATENCY_MS
from router.actions import RouterAction


def _action():
    return RouterAction("data_analysis", "python_sandbox")


def _record(success=True, cost=0.0, latency_ms=0) -> StepRecord:
    return StepRecord(
        step_id="1", workflow_id="wf-test", timestamp=0.0,
        state_features={},
        planner_suggestion=_action(), chosen_action=_action(),
        agent_name="data_analysis", tool_name="python_sandbox",
        tool_input={}, tool_output={},
        success=success, error=None,
        latency_ms=latency_ms,
        input_tokens=0, output_tokens=0,
        cost_estimate_usd=cost,
    )


def _outcome(critic_approved=True, records=None) -> WorkflowOutcome:
    return WorkflowOutcome(
        workflow_id="wf-test", question="test",
        success=True, critic_approved=critic_approved,
        total_latency_ms=0.0, total_cost_usd=0.0,
        step_records=records or [],
    )


# ---------------------------------------------------------------------------
# Basic reward range
# ---------------------------------------------------------------------------

def test_perfect_step_reward_close_to_one():
    rf = RewardFunction()
    rec = _record(success=True, cost=0.0, latency_ms=0)
    out = _outcome(critic_approved=True)
    r = rf.compute_step_reward(rec, out)
    # success=1, cost=0, time=0 → r = 1.0
    assert abs(r - 1.0) < 1e-9


def test_failed_step_reward_nonpositive():
    rf = RewardFunction()
    rec = _record(success=False, cost=0.0, latency_ms=0)
    out = _outcome(critic_approved=False)
    r = rf.compute_step_reward(rec, out)
    assert r <= 0.0


def test_reward_in_valid_range():
    rf = RewardFunction()
    # Worst case: failure + max cost + max latency
    rec = _record(success=False, cost=MAX_COST_USD, latency_ms=int(MAX_LATENCY_MS))
    out = _outcome(critic_approved=False)
    r = rf.compute_step_reward(rec, out)
    assert r >= -(rf.lambda_cost + rf.lambda_time)
    assert r <= 1.0


# ---------------------------------------------------------------------------
# Breakdown dict
# ---------------------------------------------------------------------------

def test_reward_populates_local_components():
    rf = RewardFunction()
    rec = _record(success=True, cost=0.05, latency_ms=15_000)
    out = _outcome(critic_approved=True)
    rf.compute_step_reward(rec, out)
    assert "success" in rec.local_reward_components
    assert "cost"    in rec.local_reward_components
    assert "time"    in rec.local_reward_components
    assert "reward"  in rec.local_reward_components


def test_reward_components_match_return_value():
    rf = RewardFunction()
    rec = _record(success=True, cost=0.02, latency_ms=5_000)
    out = _outcome(critic_approved=True)
    r = rf.compute_step_reward(rec, out)
    assert abs(rec.local_reward_components["reward"] - r) < 1e-12


# ---------------------------------------------------------------------------
# Normalization clamping
# ---------------------------------------------------------------------------

def test_cost_clamped_at_one():
    rf = RewardFunction()
    # cost > MAX → cost_norm = 1.0
    rec = _record(success=True, cost=MAX_COST_USD * 5, latency_ms=0)
    out = _outcome(critic_approved=True)
    rf.compute_step_reward(rec, out)
    assert rec.local_reward_components["cost"] == 1.0


def test_time_clamped_at_one():
    rf = RewardFunction()
    rec = _record(success=True, cost=0.0, latency_ms=int(MAX_LATENCY_MS * 3))
    out = _outcome(critic_approved=True)
    rf.compute_step_reward(rec, out)
    assert rec.local_reward_components["time"] == 1.0


# ---------------------------------------------------------------------------
# Lambda weights
# ---------------------------------------------------------------------------

def test_lambda_cost_scales_penalty():
    rf_low  = RewardFunction(lambda_cost=0.0,  lambda_time=0.0)
    rf_high = RewardFunction(lambda_cost=0.5,  lambda_time=0.0)
    rec_low  = _record(success=True, cost=0.05, latency_ms=0)
    rec_high = _record(success=True, cost=0.05, latency_ms=0)
    out = _outcome(critic_approved=True)
    r_low  = rf_low.compute_step_reward(rec_low,  out)
    r_high = rf_high.compute_step_reward(rec_high, out)
    assert r_low > r_high


# ---------------------------------------------------------------------------
# compute_workflow_rewards
# ---------------------------------------------------------------------------

def test_workflow_rewards_length_matches_records():
    rf = RewardFunction()
    records = [_record(success=True) for _ in range(3)]
    out = _outcome(critic_approved=True, records=records)
    rewards = rf.compute_workflow_rewards(out)
    assert len(rewards) == 3
