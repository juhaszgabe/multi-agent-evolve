"""
LangGraph orchestrator for the vanilla multi-agent data analysis system.

Graph flow:
  START → init → find_next_step → execute_step → find_next_step → ... → END

- find_next_step: marks the just-completed step as done, then picks the next
  step whose dependencies are all satisfied. If none remain, routes to END.
- execute_step: runs the appropriate agent (DataAnalyst / Visualizer / Critic /
  Writer) and updates state. When Critic says needs_revision, it un-completes
  the target step so it gets re-queued (up to MAX_CRITIC_RETRIES times).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.critic import CriticAgent
from agents.data_analyst import DataAnalystAgent
from agents.planner import PlannerAgent
from agents.visualizer import VisualizerAgent
from agents.writer import WriterAgent
from ai_providers.ai_provider import AIProvider
from tools.schema_inspector import schema_inspector

MAX_CRITIC_RETRIES = 2


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SystemState(TypedDict):
    question: str
    csv_path: str
    output_dir: str
    schema: dict
    plan: dict
    # step_id (str) → result dict
    step_results: dict
    # step IDs (int) that have finished
    completed_step_ids: list
    # step ID currently being executed (None before first step)
    current_step_id: Optional[int]
    # per-step retry counts for Critic-triggered re-runs
    retry_counts: dict
    # collected chart file paths
    chart_paths: list
    # final Writer output
    final_report: Optional[dict]


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _steps(state: SystemState) -> list:
    return state["plan"]["steps"]


def _step_by_id(state: SystemState, step_id: int) -> dict:
    return next(s for s in _steps(state) if s["id"] == step_id)


def _ready_steps(state: SystemState) -> list[dict]:
    completed = set(state["completed_step_ids"])
    return [
        s for s in _steps(state)
        if s["id"] not in completed
        and all(d in completed for d in s.get("depends_on", []))
    ]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def make_init_node(planner: PlannerAgent):
    def init_node(state: SystemState) -> dict:
        schema_res = schema_inspector(state["csv_path"])
        schema = schema_res.output if schema_res.success else {}
        plan = planner.plan(state["question"], state["csv_path"])
        print(f"[Planner] Plan:\n{json.dumps(plan, indent=2)}")
        return {
            "schema": schema,
            "plan": plan,
            "step_results": {},
            "completed_step_ids": [],
            "current_step_id": None,
            "retry_counts": {},
            "chart_paths": [],
            "final_report": None,
        }
    return init_node


def make_find_next_node():
    def find_next_node(state: SystemState) -> dict:
        completed = list(state["completed_step_ids"])

        # Mark the step that just ran as complete (if any)
        prev_id = state["current_step_id"]
        if prev_id is not None and prev_id not in completed:
            completed = completed + [prev_id]

        ready = [
            s for s in _steps(state)
            if s["id"] not in completed
            and all(d in completed for d in s.get("depends_on", []))
        ]

        next_id = ready[0]["id"] if ready else None
        print(f"[Orchestrator] Completed: {completed} | Next step: {next_id}")
        return {"completed_step_ids": completed, "current_step_id": next_id}
    return find_next_node


def make_execute_node(
    analyst: DataAnalystAgent,
    visualizer: VisualizerAgent,
    critic: CriticAgent,
    writer: WriterAgent,
):
    def execute_node(state: SystemState) -> dict:
        step_id = state["current_step_id"]
        step = _step_by_id(state, step_id)
        agent_name = step["agent"]
        task = step["task"]
        sid = str(step_id)
        print(f"[{agent_name}] Running step {step_id}: {task}")

        # Gather results from dependency steps
        prior = {
            str(d): state["step_results"].get(str(d))
            for d in step.get("depends_on", [])
        }

        # ── DataAnalyst ──────────────────────────────────────────────────────
        if agent_name == "DataAnalyst":
            result = analyst.analyze(
                task=task,
                csv_path=state["csv_path"],
                schema=state["schema"],
                prior_results=prior if prior else None,
            )
            print(f"[DataAnalyst] Success={result['success']} attempts={result['attempts']}")
            return {"step_results": {**state["step_results"], sid: result}}

        # ── Visualizer ───────────────────────────────────────────────────────
        elif agent_name == "Visualizer":
            # Use the first DataAnalyst dependency result as the data source
            da_result = next(
                (state["step_results"].get(str(d)) for d in step.get("depends_on", [])
                 if _step_by_id(state, d)["agent"] == "DataAnalyst"),
                {},
            )
            result = visualizer.visualize(
                task=task,
                analyst_result=da_result,
                output_dir=state["output_dir"],
            )
            print(f"[Visualizer] Success={result['success']} path={result.get('image_path')}")
            chart_paths = list(state["chart_paths"])
            if result.get("image_path"):
                chart_paths.append(result["image_path"])
            return {
                "step_results": {**state["step_results"], sid: result},
                "chart_paths": chart_paths,
            }

        # ── Critic ───────────────────────────────────────────────────────────
        elif agent_name == "Critic":
            da_result = None
            viz_result = None
            for dep_id in step.get("depends_on", []):
                dep_step = _step_by_id(state, dep_id)
                dep_res = state["step_results"].get(str(dep_id))
                if dep_step["agent"] == "DataAnalyst":
                    da_result = dep_res
                elif dep_step["agent"] == "Visualizer":
                    viz_result = dep_res

            verdict = critic.critique(
                question=state["question"],
                plan=state["plan"],
                analyst_result=da_result or {},
                viz_result=viz_result,
            )
            print(f"[Critic] Verdict={verdict['verdict']} issues={verdict.get('issues')}")

            step_results = {**state["step_results"], sid: verdict}
            completed = list(state["completed_step_ids"])
            retry_counts = {**state["retry_counts"]}

            if verdict["verdict"] == "needs_revision":
                action = verdict.get("suggested_action", "")
                target_agent = "DataAnalyst" if action == "rerun_data_analyst" else None

                for dep_id in step.get("depends_on", []):
                    dep_step = _step_by_id(state, dep_id)
                    if target_agent and dep_step["agent"] != target_agent:
                        continue
                    retry_key = str(dep_id)
                    retries = retry_counts.get(retry_key, 0)
                    if retries < MAX_CRITIC_RETRIES:
                        retry_counts[retry_key] = retries + 1
                        # Un-complete the target step and this Critic step
                        if dep_id in completed:
                            completed.remove(dep_id)
                        if step_id in completed:
                            completed.remove(step_id)
                        # Remove stale results
                        step_results.pop(str(dep_id), None)
                        step_results.pop(sid, None)
                        print(f"[Critic] Queuing re-run of step {dep_id} (retry {retries + 1})")
                        break

            return {
                "step_results": step_results,
                "completed_step_ids": completed,
                "retry_counts": retry_counts,
                "current_step_id": None,  # find_next will re-set this
            }

        # ── Writer ───────────────────────────────────────────────────────────
        elif agent_name == "Writer":
            result = writer.write(
                question=state["question"],
                plan=state["plan"],
                step_results=state["step_results"],
                chart_paths=state["chart_paths"],
            )
            print("[Writer] Report generated.")
            return {
                "step_results": {**state["step_results"], sid: result},
                "final_report": result,
            }

        return {}

    return execute_node


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_find_next(state: SystemState) -> str:
    return "execute_step" if state["current_step_id"] is not None else END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    provider: AIProvider,
    model_sonnet: str,
    model_haiku: str,
    output_dir: str = "outputs",
):
    planner = PlannerAgent(provider, model_sonnet)
    analyst = DataAnalystAgent(provider, model_sonnet)
    visualizer = VisualizerAgent(provider, model_haiku)
    critic = CriticAgent(provider, model_sonnet)
    writer = WriterAgent(provider, model_sonnet)

    os.makedirs(output_dir, exist_ok=True)

    graph = StateGraph(SystemState)
    graph.add_node("init", make_init_node(planner))
    graph.add_node("find_next_step", make_find_next_node())
    graph.add_node("execute_step", make_execute_node(analyst, visualizer, critic, writer))

    graph.add_edge(START, "init")
    graph.add_edge("init", "find_next_step")
    graph.add_conditional_edges("find_next_step", route_after_find_next, {"execute_step": "execute_step", END: END})
    graph.add_edge("execute_step", "find_next_step")

    return graph.compile()


def run(
    question: str,
    csv_path: str,
    provider: AIProvider,
    model_sonnet: str,
    model_haiku: str,
    output_dir: str = "outputs",
) -> dict:
    app = build_graph(provider, model_sonnet, model_haiku, output_dir)
    initial_state: SystemState = {
        "question": question,
        "csv_path": os.path.abspath(csv_path),
        "output_dir": output_dir,
        "schema": {},
        "plan": {},
        "step_results": {},
        "completed_step_ids": [],
        "current_step_id": None,
        "retry_counts": {},
        "chart_paths": [],
        "final_report": None,
    }
    final_state = app.invoke(initial_state)
    return final_state
