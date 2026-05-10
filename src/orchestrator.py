"""
LangGraph orchestrator for the multi-agent data analysis system.

Graph flow:
  START → init → find_next_step → execute_step → find_next_step → ... → finalize → END

- init: runs schema_inspector + PlannerAgent, seeds RNG, initialises state.
- find_next_step: marks just-completed step done, picks next ready step.
- execute_step: dispatches to the correct agent via ROLE_TO_AGENT; calls Router
  before every dispatch; emits a StepRecord; handles skip actions.
- finalize: computes WorkflowOutcome, calls RewardFunction, calls router.update
  for every buffered (state, action) pair, saves workflow summary.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from typing import Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.critic import CriticAgent
from agents.data_analyst import DataAnalystAgent
from agents.planner import PlannerAgent, validate_plan_step
from agents.visualizer import VisualizerAgent
from agents.writer import WriterAgent
from ai_providers.ai_provider import AIProvider
from config import Config, config_from_dict, config_to_dict
from logging_schema import (
    StepRecord,
    WorkflowOutcome,
    save_step_record,
    save_workflow_summary,
)
from reward import RewardFunction
from router.actions import RouterAction, TaskState
from router.base import Router
from router.planner_router import PlannerRouter
from state_extractor import TaskStateExtractor
from tools import TOOL_REGISTRY
from tools.schema_inspector import schema_inspector


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SystemState(TypedDict):
    # Inputs
    question: str
    csv_path: str
    output_dir: str
    log_dir: str
    # Populated by init
    workflow_id: str
    schema: dict
    plan: dict
    config: dict          # Config serialized — LangGraph requires serializable values
    # Execution state
    step_results: dict    # str(step_id) → result dict (or skip sentinel)
    completed_step_ids: list
    current_step_id: Optional[int]
    retry_counts: dict    # str(step_id) → int
    chart_paths: list
    # Logging buffers
    step_records: list    # list of StepRecord-compatible dicts
    pending_router_updates: list  # [{state_vec, action, record_index}]
    # Output
    final_report: Optional[dict]
    critic_approved: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _steps(state: SystemState) -> list:
    return state["plan"]["steps"]


def _step_by_id(state: SystemState, step_id: int) -> dict:
    return next(s for s in _steps(state) if s["id"] == step_id)


def _resolve_tool(tool_name: Optional[str]):
    if tool_name is None:
        return None
    return TOOL_REGISTRY.get(tool_name)


def _action_to_dict(action: RouterAction) -> dict:
    return {"agent_role": action.agent_role, "tool_name": action.tool_name}


def _dict_to_action(d) -> RouterAction:
    if isinstance(d, RouterAction):
        return d
    return RouterAction(agent_role=d["agent_role"], tool_name=d["tool_name"])


def _record_to_dict(record: "StepRecord") -> dict:
    d = record.__dict__.copy()
    d["planner_suggestion"] = _action_to_dict(record.planner_suggestion)
    d["chosen_action"] = _action_to_dict(record.chosen_action)
    return d


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_init_node(planner: PlannerAgent):
    def init_node(state: SystemState) -> dict:
        cfg = config_from_dict(state["config"])
        random.seed(cfg.random_seed)
        try:
            import numpy as np
            np.random.seed(cfg.random_seed)
        except ImportError:
            pass

        schema_res = schema_inspector(state["csv_path"])
        schema = schema_res.output if schema_res.success else {}
        plan = planner.plan(state["question"], state["csv_path"])
        workflow_id = str(uuid.uuid4())
        print(f"[Orchestrator] Workflow {workflow_id} | Plan: {json.dumps(plan, indent=2)}")
        return {
            "workflow_id": workflow_id,
            "schema": schema,
            "plan": plan,
            "step_results": {},
            "completed_step_ids": [],
            "current_step_id": None,
            "retry_counts": {},
            "chart_paths": [],
            "step_records": [],
            "pending_router_updates": [],
            "final_report": None,
            "critic_approved": False,
        }
    return init_node


def make_find_next_node():
    def find_next_node(state: SystemState) -> dict:
        completed = list(state["completed_step_ids"])

        ready = [
            s for s in _steps(state)
            if s["id"] not in completed
            and all(d in completed for d in s.get("depends_on", []))
        ]
        next_id = ready[0]["id"] if ready else None
        print(f"[Orchestrator] Completed: {completed} | Next: {next_id}")
        return {"completed_step_ids": completed, "current_step_id": next_id}
    return find_next_node


def make_execute_node(
    analyst: DataAnalystAgent,
    visualizer: VisualizerAgent,
    critic: CriticAgent,
    writer: WriterAgent,
    router: Router,
    extractor: TaskStateExtractor,
    reward_fn: RewardFunction,
    log_dir: str,
    skill_library=None,
    error_catalog=None,
):
    # Role → agent callable mapping — no if-elif chains
    def _dispatch(role, action, step, state):
        sid = str(step["id"])
        task = step["task"]
        tool = _resolve_tool(action.tool_name)
        prior = {str(d): state["step_results"].get(str(d)) for d in step.get("depends_on", [])}

        if role == "data_analysis":
            return analyst.analyze(
                task=task, csv_path=state["csv_path"], schema=state["schema"],
                prior_results=prior if prior else None, tool=tool,
            )
        elif role == "visualization":
            da_result = next(
                (state["step_results"].get(str(d)) for d in step.get("depends_on", [])
                 if _step_by_id(state, d)["agent_role"] == "data_analysis"),
                {},
            )
            return visualizer.visualize(
                task=task, analyst_result=da_result or {},
                output_dir=state["output_dir"], tool=tool,
            )
        elif role == "verification":
            da_result = next(
                (state["step_results"].get(str(d)) for d in step.get("depends_on", [])
                 if _step_by_id(state, d)["agent_role"] == "data_analysis"),
                None,
            )
            viz_result = next(
                (state["step_results"].get(str(d)) for d in step.get("depends_on", [])
                 if _step_by_id(state, d)["agent_role"] == "visualization"),
                None,
            )
            return critic.critique(
                question=state["question"], plan=state["plan"],
                analyst_result=da_result, viz_result=viz_result,
            )
        elif role == "synthesis":
            return writer.write(
                question=state["question"], plan=state["plan"],
                step_results=state["step_results"], chart_paths=state["chart_paths"],
            )
        return {}

    def execute_node(state: SystemState) -> dict:
        step_id = state["current_step_id"]
        step = _step_by_id(state, step_id)
        validate_plan_step(step)
        sid = str(step_id)
        role = step["agent_role"]
        cfg = config_from_dict(state["config"])

        # Build task state for router
        prior = {str(d): state["step_results"].get(str(d)) for d in step.get("depends_on", [])}
        task_state: TaskState = extractor.extract(step["task"], state["schema"], prior)

        # Planner suggestion (from plan)
        planner_suggestion = RouterAction(
            agent_role=role,
            tool_name=step.get("suggested_tool"),
        )

        # Router decision
        if isinstance(router, PlannerRouter):
            chosen_action = router.select_action(task_state, plan_step=step)
        else:
            chosen_action = router.select_action(task_state)

        agent_name = role
        print(f"[{role}] step {step_id} | tool={chosen_action.tool_name} | task={step['task'][:60]}")

        t_start = time.time()

        # --- Skip handling ---
        if chosen_action.agent_role == "skip":
            completed = list(state["completed_step_ids"])
            if step_id not in completed:
                completed.append(step_id)
            skip_result = {"skipped": True, "agent_role": role}
            step_results = {**state["step_results"], sid: skip_result}
            record = StepRecord(
                step_id=sid, workflow_id=state["workflow_id"], timestamp=t_start,
                state_features={"task_type": task_state.task_type,
                                 "has_temporal": task_state.has_temporal_keyword},
                planner_suggestion=planner_suggestion, chosen_action=chosen_action,
                agent_name=agent_name, tool_name=None,
                tool_input={}, tool_output=skip_result,
                success=True, error=None, latency_ms=0,
                input_tokens=0, output_tokens=0, cost_estimate_usd=0.0,
            )
            save_step_record(record, log_dir)
            return {
                "step_results": step_results,
                "completed_step_ids": completed,
                "step_records": state["step_records"] + [_record_to_dict(record)],
                "pending_router_updates": state["pending_router_updates"] + [
                    {"task_state": task_state.__dict__, "action": _action_to_dict(chosen_action),
                     "record_index": len(state["step_records"])}
                ],
            }

        # --- Normal execution ---
        result = _dispatch(role, chosen_action, step, state)
        latency_ms = int((time.time() - t_start) * 1000)

        success = result.get("success", True) if isinstance(result, dict) else True
        error = result.get("error") if isinstance(result, dict) else None
        in_tok = result.get("input_tokens", 0) if isinstance(result, dict) else 0
        out_tok = result.get("output_tokens", 0) if isinstance(result, dict) else 0
        cost = (in_tok * 0.8 + out_tok * 2.4) / 1_000_000  # rough estimate per token

        record = StepRecord(
            step_id=sid, workflow_id=state["workflow_id"], timestamp=t_start,
            state_features={"task_type": task_state.task_type,
                             "has_temporal": task_state.has_temporal_keyword},
            planner_suggestion=planner_suggestion, chosen_action=chosen_action,
            agent_name=agent_name, tool_name=chosen_action.tool_name,
            tool_input={"task": step["task"]},
            tool_output={k: v for k, v in (result or {}).items()
                         if k not in ("input_tokens", "output_tokens")},
            success=success, error=error, latency_ms=latency_ms,
            input_tokens=in_tok, output_tokens=out_tok, cost_estimate_usd=cost,
        )
        save_step_record(record, log_dir)

        step_results = {**state["step_results"], sid: result}
        chart_paths = list(state["chart_paths"])
        if role == "visualization" and isinstance(result, dict) and result.get("image_path"):
            chart_paths.append(result["image_path"])

        # Critic: handle needs_revision → un-complete target step
        completed = list(state["completed_step_ids"])
        if step_id not in completed:
            completed.append(step_id)
        retry_counts = {**state["retry_counts"]}
        critic_approved = state["critic_approved"]

        if role == "verification":
            verdict = result.get("verdict", "approved") if isinstance(result, dict) else "approved"
            if verdict == "approved":
                critic_approved = True
            elif verdict == "needs_revision":
                if error_catalog is not None:
                    issues = result.get("issues", []) if isinstance(result, dict) else []
                    for dep_id in step.get("depends_on", []):
                        dep_res = state["step_results"].get(str(dep_id)) or {}
                        if isinstance(dep_res, dict) and not dep_res.get("skipped"):
                            error_catalog.add(
                                task=_step_by_id(state, dep_id)["task"],
                                bad_code=dep_res.get("code", ""),
                                error_description="; ".join(issues),
                                fix=result.get("suggested_action") or "",
                                workflow_id=state["workflow_id"],
                                step_id=sid,
                            )
                action_str = result.get("suggested_action", "") if isinstance(result, dict) else ""
                target_role = "data_analysis" if action_str == "rerun_data_analyst" else None
                for dep_id in step.get("depends_on", []):
                    dep_step = _step_by_id(state, dep_id)
                    if target_role and dep_step["agent_role"] != target_role:
                        continue
                    retry_key = str(dep_id)
                    retries = retry_counts.get(retry_key, 0)
                    if retries < cfg.max_retries_critic:
                        retry_counts[retry_key] = retries + 1
                        if dep_id in completed:
                            completed.remove(dep_id)
                        if step_id in completed:
                            completed.remove(step_id)
                        step_results.pop(str(dep_id), None)
                        step_results.pop(sid, None)
                        print(f"[Critic] Re-queuing step {dep_id} (retry {retries + 1})")
                        break

        # Final report
        final_report = state["final_report"]
        if role == "synthesis" and isinstance(result, dict):
            final_report = result

        pending = state["pending_router_updates"] + [
            {"task_state": task_state.__dict__, "action": _action_to_dict(chosen_action),
             "record_index": len(state["step_records"])}
        ]

        return {
            "step_results": step_results,
            "completed_step_ids": completed,
            "retry_counts": retry_counts,
            "chart_paths": chart_paths,
            "step_records": state["step_records"] + [_record_to_dict(record)],
            "pending_router_updates": pending,
            "final_report": final_report,
            "critic_approved": critic_approved,
            "current_step_id": None,
        }

    return execute_node


def make_finalize_node(router: Router, reward_fn: RewardFunction, log_dir: str, skill_library=None):
    def finalize_node(state: SystemState) -> dict:
        total_latency = sum(r.get("latency_ms", 0) for r in state["step_records"])
        total_cost = sum(r.get("cost_estimate_usd", 0.0) for r in state["step_records"])

        # Re-hydrate StepRecords for reward computation
        import dataclasses
        records = []
        for rd in state["step_records"]:
            r = StepRecord(**{
                k: _dict_to_action(v) if k in ("planner_suggestion", "chosen_action") else v
                for k, v in rd.items()
            })
            records.append(r)

        outcome = WorkflowOutcome(
            workflow_id=state["workflow_id"],
            question=state["question"],
            success=state["final_report"] is not None,
            critic_approved=state["critic_approved"],
            total_latency_ms=total_latency,
            total_cost_usd=total_cost,
            step_records=records,
        )

        rewards = reward_fn.compute_workflow_rewards(outcome)

        # Router updates (terminal reward, not placeholder)
        for upd, reward in zip(state["pending_router_updates"], rewards):
            ts = TaskState(**upd["task_state"])
            action = _dict_to_action(upd["action"])
            router.update(ts, action, reward)

        if skill_library is not None:
            cfg = config_from_dict(state["config"])
            for record, reward in zip(records, rewards):
                if reward >= cfg.skill_write_threshold and record.success:
                    code = (record.tool_output or {}).get("code", "")
                    task = (record.tool_input or {}).get("task", "")
                    if task and code:
                        skill_library.add(
                            task=task, code=code, reward=reward,
                            workflow_id=state["workflow_id"],
                        )

        save_workflow_summary(outcome, log_dir)
        print(f"[Orchestrator] Workflow {state['workflow_id']} complete | "
              f"cost=${total_cost:.4f} | latency={total_latency}ms")
        return {}

    return finalize_node


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_find_next(state: SystemState) -> str:
    return "execute_step" if state["current_step_id"] is not None else "finalize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    provider: AIProvider,
    config: Config,
    output_dir: str = "outputs",
    log_dir: str = "logs",
    router: Router | None = None,
):
    planner = PlannerAgent(provider, config.model_per_role["planner"], config)
    visualizer = VisualizerAgent(provider, config.model_per_role["visualization"], config)
    critic = CriticAgent(provider, config.model_per_role["verification"], config)
    writer = WriterAgent(provider, config.model_per_role["synthesis"], config)

    skill_lib = None
    error_cat = None
    if config.enable_memory:
        from memory import SkillLibrary, ErrorCatalog
        mem_root = os.path.join(log_dir, "memory")
        skill_lib = SkillLibrary(persist_dir=os.path.join(mem_root, "skills"))
        error_cat = ErrorCatalog(persist_dir=os.path.join(mem_root, "errors"))

    analyst = DataAnalystAgent(
        provider, config.model_per_role["data_analysis"], config,
        skill_library=skill_lib, error_catalog=error_cat,
    )

    effective_router: Router = router or PlannerRouter()
    extractor = TaskStateExtractor()
    reward_fn = RewardFunction(lambda_cost=config.lambda_cost, lambda_time=config.lambda_time)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    graph = StateGraph(SystemState)
    graph.add_node("init", make_init_node(planner))
    graph.add_node("find_next_step", make_find_next_node())
    graph.add_node("execute_step", make_execute_node(
        analyst, visualizer, critic, writer,
        effective_router, extractor, reward_fn, log_dir,
        skill_library=skill_lib, error_catalog=error_cat,
    ))
    graph.add_node("finalize", make_finalize_node(effective_router, reward_fn, log_dir,
                                                   skill_library=skill_lib))

    graph.add_edge(START, "init")
    graph.add_edge("init", "find_next_step")
    graph.add_conditional_edges(
        "find_next_step", route_after_find_next,
        {"execute_step": "execute_step", "finalize": "finalize"},
    )
    graph.add_edge("execute_step", "find_next_step")
    graph.add_edge("finalize", END)

    return graph.compile()


def run(
    question: str,
    csv_path: str,
    provider: AIProvider,
    config: Config,
    output_dir: str = "outputs",
    log_dir: str = "logs",
    router: Router | None = None,
) -> dict:
    app = build_graph(provider, config, output_dir, log_dir, router)
    initial_state: SystemState = {
        "question": question,
        "csv_path": os.path.abspath(csv_path),
        "output_dir": output_dir,
        "log_dir": log_dir,
        "workflow_id": "",
        "schema": {},
        "plan": {},
        "config": config_to_dict(config),
        "step_results": {},
        "completed_step_ids": [],
        "current_step_id": None,
        "retry_counts": {},
        "chart_paths": [],
        "step_records": [],
        "pending_router_updates": [],
        "final_report": None,
        "critic_approved": False,
    }
    return app.invoke(initial_state)
